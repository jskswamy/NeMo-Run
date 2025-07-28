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

from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.execution.kubeflow import KubeflowExecutor
from nemo_run.core.packaging.configmap import ConfigMapPackager


def test_kubeflow_executor_default_init():
    """Test that KubeflowExecutor initializes with default values."""
    executor = KubeflowExecutor()

    assert executor.nodes == 1
    assert executor.ntasks_per_node == 1
    assert executor.namespace == "default"
    assert executor.python_file is None
    assert executor.gpus == 1
    assert executor.runtime_name == "torch-distributed-nemo"
    assert executor.job_name == ""  # Should start empty
    assert isinstance(executor.packager, ConfigMapPackager)


def test_kubeflow_executor_custom_init():
    """Test that KubeflowExecutor initializes with custom values."""
    executor = KubeflowExecutor(
        nodes=2,
        ntasks_per_node=4,
        namespace="training",
        python_file="train.py",
        gpus=8,
        runtime_name="custom-runtime",
    )

    assert executor.nodes == 2
    assert executor.ntasks_per_node == 4
    assert executor.namespace == "training"
    assert executor.python_file == "train.py"
    assert executor.gpus == 8
    assert executor.runtime_name == "custom-runtime"


def test_kubeflow_executor_assign():
    """Test that assign method sets the correct directories."""
    executor = KubeflowExecutor()
    executor.assign("exp-123", "/tmp/exp", "task-1", "task_dir")

    assert executor.experiment_id == "exp-123"
    assert executor.experiment_dir == "/tmp/exp"
    assert executor.job_dir == "/tmp/exp/task_dir"
    assert executor.job_name == "task-1"


def test_kubeflow_executor_nnodes():
    """Test that nnodes returns the correct number of nodes."""
    executor = KubeflowExecutor(nodes=3)
    assert executor.nnodes() == 3


def test_kubeflow_executor_nproc_per_node():
    """Test that nproc_per_node returns the correct number of processes."""
    executor = KubeflowExecutor(ntasks_per_node=4)
    assert executor.nproc_per_node() == 4


def test_kubeflow_executor_get_runtime():
    """Test that _get_runtime returns the correct Runtime configuration."""
    executor = KubeflowExecutor(python_file="train.py", gpus=4, runtime_name="custom-runtime")
    runtime = executor._get_runtime()

    assert runtime.name == "custom-runtime"
    assert runtime.trainer is not None
    assert runtime.trainer.framework.value == "torch"
    assert runtime.trainer.accelerator == "gpu"
    assert runtime.trainer.accelerator_count == 4


def test_kubeflow_executor_get_custom_trainer_file_based():
    """Test that _get_custom_trainer returns correct configuration for file-based execution."""
    executor = KubeflowExecutor(
        python_file="train.py",
        nodes=2,
        gpus=8,
        cpu_request="8",
        cpu_limit="16",
        memory_request="16Gi",
        memory_limit="32Gi",
    )

    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_custom_trainer:
        mock_trainer = MagicMock()
        mock_custom_trainer.return_value = mock_trainer

        trainer = executor._get_custom_trainer()

        # Verify CustomTrainer was called with correct arguments
        mock_custom_trainer.assert_called_once()
        call_args = mock_custom_trainer.call_args[1]
        assert call_args["python_file"] == "train.py"
        assert "func" not in call_args
        assert call_args["num_nodes"] == 2
        assert call_args["resources_per_node"] is not None


def test_kubeflow_executor_get_custom_trainer_function_based():
    """Test that _get_custom_trainer returns correct configuration for function-based execution."""

    def dummy_function():
        pass

    executor = KubeflowExecutor(nodes=1, gpus=1, func=dummy_function)

    with patch("nemo_run.core.execution.kubeflow.CustomTrainer") as mock_custom_trainer:
        mock_trainer = MagicMock()
        mock_custom_trainer.return_value = mock_trainer

        trainer = executor._get_custom_trainer()

        # Verify CustomTrainer was called with correct arguments
        mock_custom_trainer.assert_called_once()
        call_args = mock_custom_trainer.call_args[1]
        assert "python_file" not in call_args
        assert "func" in call_args
        assert call_args["func"] == dummy_function
        assert call_args["num_nodes"] == 1
        assert call_args["resources_per_node"] is not None


def test_kubeflow_executor_create_trainjob():
    """Test that create_trainjob uses the SDK correctly."""
    executor = KubeflowExecutor(python_file="train.py")
    executor.assign("exp-123", "/tmp/exp", "my-task", "task_dir")

    with patch.object(executor, "_get_trainer_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.train.return_value = "job-123"
        mock_get_client.return_value = mock_client

        with patch.object(executor, "stage_files") as mock_stage:
            mock_stage.return_value = "configmap-name"

            job_id = executor.create_trainjob("test-job")

            assert job_id == "job-123"
            mock_client.train.assert_called_once()
            mock_stage.assert_called_once_with("task_dir")


def test_kubeflow_executor_get_trainjob_status():
    """Test that get_trainjob_status works correctly."""
    executor = KubeflowExecutor(python_file="train.py")

    with patch.object(executor, "_get_trainer_client") as mock_get_client:
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.status = "Running"
        mock_client.get_job.return_value = mock_job
        mock_get_client.return_value = mock_client

        status = executor.get_trainjob_status("job-123")

        assert status == "Running"
        mock_client.get_job.assert_called_once_with("job-123")


def test_kubeflow_executor_delete_trainjob():
    """Test that delete_trainjob uses the SDK correctly."""
    executor = KubeflowExecutor()

    with patch.object(executor, "_get_trainer_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        executor.delete_trainjob("job-123")

        mock_client.delete_job.assert_called_once_with("job-123")


def test_kubeflow_executor_get_trainjob_logs():
    """Test that get_trainjob_logs uses the SDK correctly."""
    executor = KubeflowExecutor()

    with patch.object(executor, "_get_trainer_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = {"logs": "test logs"}
        mock_get_client.return_value = mock_client

        logs = executor.get_trainjob_logs("job-123", follow=True)

        assert logs == {"logs": "test logs"}
        mock_client.get_job_logs.assert_called_once_with("job-123", follow=True)


@pytest.mark.parametrize(
    "executor_kwargs,expected_mode,expected_nodes,expected_gpus",
    [
        ({"python_file": "train.py", "nodes": 2, "gpus": 4}, "file-based", 2, 4),
        ({"nodes": 1, "gpus": 1}, "function-based", 1, 1),
    ],
)
def test_kubeflow_executor_info(executor_kwargs, expected_mode, expected_nodes, expected_gpus):
    """Test that info method returns correct information for different execution modes."""
    executor = KubeflowExecutor(**executor_kwargs)
    info = executor.info()
    expected_info = (
        f"KubeflowExecutor({expected_mode}, nodes={expected_nodes}, gpus={expected_gpus})"
    )
    assert expected_info in info


def test_kubeflow_executor_stage_files():
    """Test that stage_files uses ConfigMapPackager correctly."""
    executor = KubeflowExecutor()
    executor.experiment_id = "exp-123"
    executor.experiment_dir = "/tmp/exp"

    with patch.object(executor.packager, "package") as mock_package:
        mock_package.return_value = "configmap-name"

        result = executor.stage_files("task_dir")

        # Verify the package method was called with correct arguments
        mock_package.assert_called_once()
        call_args = mock_package.call_args
        assert call_args[1]["job_dir"] == "task_dir"
        assert call_args[1]["name"] == "exp-123-task_dir"
        assert result == "configmap-name"


def test_kubeflow_executor_cleanup_files():
    """Test that cleanup_files uses ConfigMapPackager correctly."""
    executor = KubeflowExecutor()
    executor.experiment_id = "exp-123"

    with patch.object(executor.packager, "cleanup") as mock_cleanup:
        executor.cleanup_files("task_dir")

        mock_cleanup.assert_called_once_with("exp-123-task_dir")
