# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.types.types import CustomTrainer, Framework, Runtime, Trainer, TrainerType

from nemo_run.core.execution.base import Executor
from nemo_run.core.packaging.configmap import ConfigMapPackager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KubeflowExecutor(Executor):
    """
    Dataclass to configure Kubeflow executor for distributed training jobs.

    This executor uses the Kubeflow Trainer SDK to create and manage TrainJob objects.
    It supports both file-based and function-based execution modes.
    For file-based execution, it stages files into Kubernetes ConfigMaps.
    For function-based execution, it serializes functions and stages them as well.

    The actual execution details (torchrun vs python, command construction) are handled
    by the Kubeflow SDK through the Runtime and Trainer objects.

    Example:

    .. code-block:: python

        # File-based execution
        executor = KubeflowExecutor(
            packager=ConfigMapPackager(include_pattern="*.py"),
            python_file="train.py",
            namespace="default"
        )

        # Or use function-based execution
        def my_training_function():
            import torch
            print("Training with PyTorch...")
            # Your training logic here

        executor = KubeflowExecutor(
            packager=ConfigMapPackager(include_pattern="*.py"),
            func=my_training_function,
            namespace="default"
        )

        # Example: specifying a custom ClusterTrainingRuntime by name
        executor = KubeflowExecutor(
            packager=ConfigMapPackager(include_pattern="*.py"),
            python_file="train.py",
            namespace="default",
            runtime_name="my-custom-clusterruntime"
        )
    """

    #: Number of nodes for distributed training
    nodes: int = 1

    #: Number of processes per node (typically matches number of GPUs)
    ntasks_per_node: int = 1

    #: Kubernetes namespace for the training job
    namespace: str = "default"

    #: Python file to execute (for file-based execution)
    python_file: Optional[str] = None

    #: Function to execute (for function-based execution)
    func: Optional[Callable] = None

    #: Resource requests for CPU
    cpu_request: str = "4"

    #: Resource limits for CPU
    cpu_limit: str = "8"

    #: Resource requests for memory
    memory_request: str = "8Gi"

    #: Resource limits for memory
    memory_limit: str = "16Gi"

    #: Number of GPUs to request
    gpus: int = 1

    #: Name of the ClusterTrainingRuntime to use
    runtime_name: str = "torch-distributed-nemo"

    #: TrainerClient instance for managing TrainJob objects
    _trainer_client: Optional[TrainerClient] = None

    #: Job name (set from task_id during assign)
    job_name: str = field(init=False, default="")

    def __post_init__(self):
        """Initialize the executor with ConfigMapPackager if not provided."""
        if not isinstance(self.packager, ConfigMapPackager):
            # Use ConfigMapPackager as default packager
            self.packager = ConfigMapPackager(
                include_pattern="*.py", relative_path=".", namespace=self.namespace
            )

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        """Assign experiment and task directories to the executor."""
        self.experiment_id = exp_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)
        self.job_name = task_id

    def nnodes(self) -> int:
        """Return the number of nodes for distributed training."""
        return self.nodes

    def nproc_per_node(self) -> int:
        """Return the number of processes per node."""
        return self.ntasks_per_node

    def _get_trainer_client(self) -> TrainerClient:
        """Get or create the TrainerClient instance."""
        if self._trainer_client is None:
            self._trainer_client = TrainerClient(namespace=self.namespace)
        return self._trainer_client

    def _get_runtime(self) -> Runtime:
        """Get the Runtime configuration for the training job."""
        # Create a basic runtime configuration
        # The entrypoint will be determined by the ClusterTrainingRuntime
        # We don't need to manually set it here
        trainer = Trainer(
            trainer_type=TrainerType.CUSTOM_TRAINER,
            framework=Framework.TORCH,
            # Let the ClusterTrainingRuntime determine the entrypoint
            accelerator="gpu" if self.gpus > 0 else "cpu",
            accelerator_count=self.gpus,
        )

        return Runtime(name=self.runtime_name, trainer=trainer)

    def _get_custom_trainer(self) -> CustomTrainer:
        """Get the CustomTrainer configuration for the training job."""
        resources_per_node = {
            "limits": {
                "cpu": self.cpu_limit,
                "memory": self.memory_limit,
                "nvidia.com/gpu": str(self.gpus),
            },
            "requests": {
                "cpu": self.cpu_request,
                "memory": self.memory_request,
                "nvidia.com/gpu": str(self.gpus),
            },
        }

        # Create CustomTrainer with either python_file or func
        trainer_kwargs = {"num_nodes": self.nodes, "resources_per_node": resources_per_node}

        if self.python_file:
            trainer_kwargs["python_file"] = self.python_file
        elif self.func:
            trainer_kwargs["func"] = self.func
        else:
            raise ValueError("Either python_file or func must be specified")

        return CustomTrainer(**trainer_kwargs)

    def create_trainjob(self, job_name: str) -> str:
        """Create a TrainJob using the Kubeflow SDK."""
        try:
            client = self._get_trainer_client()
            runtime = self._get_runtime()
            trainer = self._get_custom_trainer()

            # Stage files if using ConfigMapPackager
            if isinstance(self.packager, ConfigMapPackager):
                configmap_name = self.stage_files("task_dir")
                logger.info(f"Staged files in ConfigMap: {configmap_name}")

            # TODO: Use job_name once Kubeflow SDK supports custom job names
            # Currently the SDK generates random names, but we store job_name for future use
            # when the SDK adds support for custom job names
            job_id = client.train(runtime=runtime, trainer=trainer)

            logger.info(f"Created TrainJob: {job_id}")
            return job_id

        except Exception as e:
            logger.error(f"Failed to create TrainJob: {e}")
            raise

    def get_trainjob_status(self, job_name: str) -> str:
        """Get the status of a TrainJob."""
        try:
            client = self._get_trainer_client()
            job = client.get_job(job_name)
            return job.status or "Unknown"
        except Exception as e:
            logger.error(f"Failed to get TrainJob status: {e}")
            return "Unknown"

    def delete_trainjob(self, job_name: str):
        """Delete a TrainJob."""
        try:
            client = self._get_trainer_client()
            client.delete_job(job_name)
            logger.info(f"Deleted TrainJob: {job_name}")
        except Exception as e:
            logger.error(f"Failed to delete TrainJob: {e}")

    def get_trainjob_logs(self, job_name: str, follow: bool = False) -> dict:
        """Get logs from a TrainJob."""
        try:
            client = self._get_trainer_client()
            return client.get_job_logs(job_name, follow=follow)
        except Exception as e:
            logger.error(f"Failed to get TrainJob logs: {e}")
            return {}

    def stage_files(self, task_dir: str) -> str:
        """Stage files using the ConfigMapPackager."""
        if isinstance(self.packager, ConfigMapPackager):
            return self.packager.package(
                path=Path(self.experiment_dir),
                job_dir=task_dir,
                name=f"{self.experiment_id}-{task_dir}",
            )
        else:
            logger.warning("Non-ConfigMapPackager used, file staging may not work as expected")
            return ""

    def cleanup_files(self, task_dir: str):
        """Clean up staged files."""
        if isinstance(self.packager, ConfigMapPackager):
            self.packager.cleanup(f"{self.experiment_id}-{task_dir}")

    def info(self) -> str:
        """Return information about this executor."""
        mode = "file-based" if self.python_file else "function-based"
        return f"KubeflowExecutor({mode}, nodes={self.nodes}, gpus={self.gpus})"
