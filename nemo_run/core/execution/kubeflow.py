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
from typing import Optional, Union

from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.types.types import CustomTrainer, Runtime

from nemo_run.config import Partial, Script
from nemo_run.core.execution.base import Executor
from nemo_run.core.packaging.base import sanitize_kubernetes_name
from nemo_run.core.packaging.configmap import ConfigMapPackager

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KubeflowExecutor(Executor):
    """
    Dataclass to configure Kubeflow executor for distributed training jobs.

    This executor uses the Kubeflow Trainer SDK to create and manage TrainJob objects.
    It supports execution of tasks passed from the Experiment API (Script, Partial, Config).

    The actual execution details (torchrun vs python, command construction) are handled
    by the Kubeflow SDK through the Runtime and Trainer objects.

    Example:

    .. code-block:: python

        # Configure executor for execution environment
        executor = KubeflowExecutor(
            namespace="default",
            runtime_name="torch-distributed-nemo"
        )

        # Use with Experiment API
        training_script = run.Script(inline="python train.py")

        with run.Experiment("training") as exp:
            exp.add(training_script, executor=executor)
            exp.run()
    """

    #: Number of nodes for distributed training
    nodes: int = 1

    #: Number of processes per node (typically matches number of GPUs)
    ntasks_per_node: int = 1

    #: Kubernetes namespace for the training job
    namespace: str = "default"

    #: Resource requests for CPU (optional - defaults to ClusterTrainingRuntime)
    cpu_request: Optional[str] = None

    #: Resource limits for CPU (optional - defaults to ClusterTrainingRuntime)
    cpu_limit: Optional[str] = None

    #: Resource requests for memory (optional - defaults to ClusterTrainingRuntime)
    memory_request: Optional[str] = None

    #: Resource limits for memory (optional - defaults to ClusterTrainingRuntime)
    memory_limit: Optional[str] = None

    #: Number of GPUs to request (optional - defaults to ClusterTrainingRuntime)
    gpus: Optional[int] = None

    #: Name of the ClusterTrainingRuntime to use
    runtime_name: str = "torch-distributed-nemo"

    #: Volume mount path for staged files (default: /workspace)
    volume_mount_path: str = "/workspace"

    #: Default task directory name (default: "task-dir")
    default_task_dir: str = "task-dir"

    #: TrainerClient instance for managing TrainJob objects
    _trainer_client: Optional[TrainerClient] = None

    #: Job name (set from task_id during assign)
    job_name: str = field(init=False, default="")

    #: Current task being executed (set by Experiment API)
    _current_task: Optional[Union[Script, Partial]] = None

    def __post_init__(self):
        """Validate executor configuration."""
        if self.nodes < 1:
            raise ValueError("nodes must be >= 1")
        if self.ntasks_per_node < 1:
            raise ValueError("ntasks_per_node must be >= 1")

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        """Assign experiment and task information to the executor."""
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
        """Get or create a TrainerClient instance."""
        if self._trainer_client is None:
            self._trainer_client = TrainerClient()
        return self._trainer_client

    def _get_runtime(self) -> Runtime:
        """Get the Runtime configuration for the training job."""
        return Runtime(
            name=self.runtime_name,
        )

    def _get_custom_trainer(self, task) -> CustomTrainer:
        """Get the CustomTrainer configuration for the training job."""
        # Create CustomTrainer with task from Experiment API
        trainer_kwargs: dict = {"num_nodes": self.nodes}

        # Set resources - explicitly empty if not specified to override SDK defaults
        resources_per_node: dict = {}
        if self.cpu_limit is not None:
            resources_per_node["cpu"] = self.cpu_limit
        if self.memory_limit is not None:
            resources_per_node["memory"] = self.memory_limit
        if self.gpus is not None:
            resources_per_node["nvidia.com/gpu"] = str(self.gpus)

        # Always set resources_per_node to override SDK defaults
        # If empty, it will result in no resource limits
        trainer_kwargs["resources_per_node"] = resources_per_node

        # Handle task from Experiment API
        if hasattr(task, "inline") and task.inline:  # Script object
            trainer_kwargs["python_file"] = task.inline
        elif hasattr(task, "__fn_or_cls__"):  # Partial object
            trainer_kwargs["func"] = task.__fn_or_cls__
        else:
            raise ValueError("Task must be a Script or Partial object")

        return CustomTrainer(**trainer_kwargs)

    def _get_staged_file_path(self, filename: str) -> str:
        """
        Infer the correct path to a staged file based on how it was staged.

        This method determines the full path to a staged file by:
        1. Getting the expected file path from the ConfigMapPackager
        2. Using the volume mount path from the ClusterTrainingRuntime

        Args:
            filename: The filename to resolve (e.g., "mistral.py")

        Returns:
            The full path to the staged file in the container
        """
        # Get the task directory from job_dir if available
        task_dir = self.default_task_dir  # Use the configurable default
        if hasattr(self, "job_dir") and self.job_dir:
            task_dir = os.path.basename(self.job_dir)

        # Determine the file path based on the packager
        if isinstance(self.packager, ConfigMapPackager):
            # Get the expected file path from the ConfigMapPackager
            full_path = self.packager.get_container_file_path(
                task_dir, filename, self.volume_mount_path
            )

            logger.debug(f"📝 Task dir: {task_dir}")
            logger.debug(f"📁 Volume mount path: {self.volume_mount_path}")
            logger.debug(f"🔗 Full path: {full_path}")

            return full_path
        else:
            # For non-ConfigMapPackager, assume the file is in the working directory
            logger.warning("Non-ConfigMapPackager used, assuming file is in working directory")
            return filename

    def create_trainjob(self, job_name: str, task) -> str:
        """Create a TrainJob using the Kubeflow SDK."""
        try:
            client = self._get_trainer_client()
            runtime = self._get_runtime()
            trainer = self._get_custom_trainer(task)

            # Stage files if using ConfigMapPackager
            if isinstance(self.packager, ConfigMapPackager):
                configmap_name = self.stage_files(self.default_task_dir)
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

    def _get_sanitized_configmap_name(self, task_dir: str) -> str:
        """Get a sanitized ConfigMap name that complies with Kubernetes naming rules."""
        sanitized_experiment_id = sanitize_kubernetes_name(self.experiment_id or "experiment")
        sanitized_task_dir = sanitize_kubernetes_name(task_dir)

        # Use the packager's configmap_prefix if available
        configmap_prefix = getattr(self.packager, "configmap_prefix", "nemo-workspace")
        if configmap_prefix:
            return f"{configmap_prefix}-{sanitized_experiment_id}-{sanitized_task_dir}"
        else:
            return f"{sanitized_experiment_id}-{sanitized_task_dir}"

    def stage_files(self, task_dir: str) -> str:
        """Stage files using the packager and return the ConfigMap name."""
        try:
            configmap_name = self._get_sanitized_configmap_name(task_dir)
            self.packager.package(
                path=Path(self.experiment_dir), job_dir=task_dir, name=configmap_name
            )
            logger.info(f"Staged files in ConfigMap: {configmap_name}")
            return configmap_name
        except Exception as e:
            logger.error(f"Failed to stage files: {e}")
            raise

    def cleanup_files(self, task_dir: str):
        """Clean up staged files."""
        try:
            configmap_name = self._get_sanitized_configmap_name(task_dir)
            # TODO: Implement ConfigMap cleanup when Kubeflow SDK supports it
            logger.info(f"Files staged in ConfigMap: {configmap_name}")
        except Exception as e:
            logger.error(f"Failed to cleanup files: {e}")

    def submit(self, task, job_name: str) -> str:
        """
        Submit a job using the Kubeflow SDK.

        This method is called by the Experiment API to submit a task for execution.
        It handles task validation, file staging, and TrainJob creation.

        Args:
            task: The task to execute (Script or Partial object)
            job_name: The name of the job to submit

        Returns:
            The job ID returned by the Kubeflow SDK

        Raises:
            RuntimeError: If executor is not assigned to an experiment
            ValueError: If task is not a valid Script or Partial object
        """
        if not hasattr(self, "experiment_id") or not self.experiment_id:
            raise RuntimeError("Executor not assigned to experiment")

        try:
            # Stage files if using ConfigMapPackager
            if isinstance(self.packager, ConfigMapPackager):
                configmap_name = self.stage_files(self.job_dir.split("/")[-1])
                logger.info(f"Staged files in ConfigMap: {configmap_name}")

            # Create TrainJob using the Kubeflow SDK
            job_id = self.create_trainjob(job_name, task)
            logger.info(f"Submitted job {job_name} with ID: {job_id}")

            return job_id

        except Exception as e:
            logger.error(f"Failed to submit job {job_name}: {e}")
            raise

    def monitor(self, job_id: str) -> str:
        """
        Monitor the status of a submitted job.

        This method is called by the Experiment API to check job status.

        Args:
            job_id: The ID of the job to monitor

        Returns:
            The current status of the job (Running, Completed, Failed, etc.)

        Raises:
            RuntimeError: If executor is not assigned to an experiment
        """
        if not hasattr(self, "experiment_id") or not self.experiment_id:
            raise RuntimeError("Executor not assigned to experiment")

        try:
            status = self.get_trainjob_status(job_id)
            logger.debug(f"Job {job_id} status: {status}")
            return status

        except Exception as e:
            logger.error(f"Failed to monitor job {job_id}: {e}")
            return "Unknown"

    def cleanup(self, handle: str) -> None:
        """
        Clean up resources associated with a job.

        This method is called by the Experiment API to clean up job resources.
        It handles TrainJob deletion and file cleanup.

        Args:
            handle: The ID of the job to clean up

        Raises:
            RuntimeError: If executor is not assigned to an experiment
        """
        if not hasattr(self, "experiment_id") or not self.experiment_id:
            raise RuntimeError("Executor not assigned to experiment")

        try:
            # Delete the TrainJob
            self.delete_trainjob(handle)

            # Clean up staged files
            task_dir = self.job_dir.split("/")[-1] if self.job_dir else self.default_task_dir
            self.cleanup_files(task_dir)

            logger.info(f"Cleaned up job {handle}")

        except Exception as e:
            logger.error(f"Failed to cleanup job {handle}: {e}")

    def info(self) -> str:
        """Get information about the executor configuration."""
        return f"KubeflowExecutor (nodes={self.nodes}, gpus={self.gpus or 0})"
