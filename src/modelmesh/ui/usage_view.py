"""Usage analytics dashboard displaying metrics aggregated from routing logs."""

from __future__ import annotations

import json
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from modelmesh.core.storage import Storage


class UsageDialog(QDialog):
    """Analytics dialog displaying aggregated routing statistics and token usage."""

    def __init__(self, storage: Storage, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("usageDialog")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle("ModelMesh Usage & Analytics Dashboard")
        self.resize(780, 480)

        self.storage = storage

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header Summary Cards
        summary_card = QWidget()
        summary_card.setObjectName("usageSummaryCard")
        summary_layout = QHBoxLayout(summary_card)

        self.reqs_label = QLabel("<strong>Requests:</strong> 0")
        self.tokens_label = QLabel("<strong>Total Tokens:</strong> 0")
        self.cost_label = QLabel("<strong>Est. Total Cost:</strong> $0.0000")
        self.hits_label = QLabel("<strong>Cache Hits:</strong> 0 · <strong>Saved Cost:</strong> $0.0000")

        summary_layout.addWidget(self.reqs_label)
        summary_layout.addWidget(self.tokens_label)
        summary_layout.addWidget(self.cost_label)
        summary_layout.addWidget(self.hits_label)
        layout.addWidget(summary_card)

        # Usage Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Model",
            "Endpoint",
            "Provider",
            "Strategy",
            "Tokens In/Out",
            "Est. Cost",
            "Latency",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        # Bottom Buttons
        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.refresh()

    def refresh(self) -> None:
        """Fetch latest routing logs and populate table."""
        logs = self.storage.get_routing_logs(limit=100)
        self.table.setRowCount(0)

        total_reqs = 0
        total_toks = 0
        total_cost = 0.0
        total_hits = 0
        total_saved = 0.0

        for entry in logs:
            if entry.get("cache_status") == "hit":
                # Cache hits carry the saved tokens/cost, not real spend; they
                # are excluded from the real totals and summarised separately.
                total_hits += 1
                block = self._cache_block(entry)
                total_saved += float(block.get("saved_cost_usd") or 0.0)
                continue
            total_reqs += 1
            tin = entry.get("tokens_in") or 0
            tout = entry.get("tokens_out") or 0
            # Column names as stored in routing_log
            cost = entry.get("estimated_cost") or 0.0
            lat = (entry.get("total_latency_ms") or 0) / 1000.0

            total_toks += tin + tout
            total_cost += cost

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(entry.get("chosen_model_id") or "-"))
            self.table.setItem(row, 1, QTableWidgetItem(entry.get("chosen_endpoint_id") or "-"))
            self.table.setItem(row, 2, QTableWidgetItem(entry.get("provider_id") or "-"))
            self.table.setItem(row, 3, QTableWidgetItem(entry.get("strategy") or "-"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{tin} / {tout}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"${cost:.5f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{lat:.2f}s"))

        self.reqs_label.setText(f"<strong>Requests:</strong> {total_reqs}")
        self.tokens_label.setText(f"<strong>Total Tokens:</strong> {total_toks:,}")
        self.cost_label.setText(f"<strong>Est. Total Cost:</strong> ${total_cost:.4f}")
        self.hits_label.setText(
            f"<strong>Cache Hits:</strong> {total_hits} · <strong>Saved Cost:</strong> ${total_saved:.4f}"
        )

    @staticmethod
    def _cache_block(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Return the ``cache`` block from a routing_log row's features_json."""
        features_json = entry.get("features_json")
        if not features_json:
            return {}
        try:
            parsed = json.loads(features_json)
        except (json.JSONDecodeError, TypeError):
            return {}
        block = parsed.get("cache") if isinstance(parsed, dict) else None
        return block if isinstance(block, dict) else {}
