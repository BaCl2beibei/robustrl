import unittest

from ftm.backend_api import (CheckpointManifest, ExecutionHandle, Lease,
                             execute_two_phase_migration)


class Checkpoints:
    def latest(self, task_id):
        return CheckpointManifest(task_id, 3, "store://checkpoint/3", 128, "abc", 42.0)


class Transport:
    def transfer(self, source, target_node):
        return source


class Executor:
    def __init__(self):
        self.operations = []

    def reserve(self, task_id, target_node, fencing_epoch):
        self.operations.append("reserve")
        return Lease(task_id, target_node, fencing_epoch, "new-owner")

    def restore(self, lease, checkpoint):
        self.operations.append("restore")
        return ExecutionHandle(lease.task_id, lease.node_id, "target-process")

    def ready(self, handle):
        self.operations.append("ready")
        return True

    def stop(self, handle, fencing_epoch):
        self.operations.append("stop")

    def release(self, lease):
        self.operations.append("release")


class Leases:
    def __init__(self, cas_result):
        self.cas_result = cas_result

    def current(self, task_id):
        return None

    def compare_and_swap(self, task_id, expected, replacement):
        return self.cas_result


class BackendApiTests(unittest.TestCase):
    def test_commit_switches_lease_before_source_cleanup(self):
        executor = Executor()
        source_lease = Lease("t1", "node-a", 1, "old-owner")
        result = execute_two_phase_migration(
            "t1", "node-b", source_lease, 2, Checkpoints(), executor,
            Leases(True), Transport(), ExecutionHandle("t1", "node-a", "source"),
        )
        self.assertTrue(result.committed)
        self.assertEqual(executor.operations,
                         ["reserve", "restore", "ready", "stop", "release"])

    def test_failed_lease_switch_cleans_up_only_target(self):
        executor = Executor()
        result = execute_two_phase_migration(
            "t1", "node-b", Lease("t1", "node-a", 1, "old-owner"), 2,
            Checkpoints(), executor, Leases(False), Transport(),
        )
        self.assertFalse(result.committed)
        self.assertIn("source lease changed", result.reason)
        self.assertEqual(executor.operations, ["reserve", "restore", "ready", "stop", "release"])


if __name__ == "__main__":
    unittest.main()
