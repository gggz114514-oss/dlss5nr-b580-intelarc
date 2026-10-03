"""Release the retained last-entry buffers when a v5 graph session closes."""
from graph_front_v5 import GraphFront as Base


class GraphFront(Base):
    def close(self):
        super().close()
        # Base entries are cleared, but last_entry also owns static inputs/output.
        # A caller may keep the closed adapter for metadata or error reporting.
        self.last_entry = None
