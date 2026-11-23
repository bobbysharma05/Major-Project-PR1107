"""Cache settings tab: response-cache toggle, sizes, semantic settings, stats, and clear."""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modelmesh.core.cache.config import (
    SemanticCacheConfig,
    load_cache_config,
    save_cache_config,
)
from modelmesh.core.storage import Storage


class CacheTab(QWidget):
    """Settings tab managing response-cache configuration and observability."""

    def __init__(
        self,
        storage: Optional[Storage] = None,
        engine: Optional[Any] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cacheTab")
        self.storage = storage
        self.engine = engine

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        cfg = load_cache_config()

        enabled_row = QHBoxLayout()
        self.enabled_cb = QCheckBox("Cache enabled")
        self.enabled_cb.setChecked(cfg.enabled)
        self.enabled_cb.toggled.connect(self.save)
        enabled_row.addWidget(self.enabled_cb)
        enabled_row.addStretch()
        layout.addLayout(enabled_row)

        ttl_row = QHBoxLayout()
        ttl_row.addWidget(QLabel("TTL (hours):"))
        self.ttl_spin = QSpinBox()
        self.ttl_spin.setRange(1, 1000000)
        self.ttl_spin.setValue(max(1, cfg.ttl_seconds // 3600))
        self.ttl_spin.valueChanged.connect(self.save)
        ttl_row.addWidget(self.ttl_spin)
        ttl_row.addStretch()
        layout.addLayout(ttl_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max entries:"))
        self.max_entries_spin = QSpinBox()
        self.max_entries_spin.setRange(1, 100000)
        self.max_entries_spin.setValue(cfg.max_entries)
        self.max_entries_spin.valueChanged.connect(self.save)
        max_row.addWidget(self.max_entries_spin)
        max_row.addStretch()
        layout.addLayout(max_row)

        semantic_row = QHBoxLayout()
        self.semantic_cb = QCheckBox("Semantic matching (near-duplicate prompts)")
        self.semantic_cb.setChecked(cfg.semantic.enabled)
        self.semantic_cb.toggled.connect(self.save)
        semantic_row.addWidget(self.semantic_cb)
        semantic_row.addStretch()
        layout.addLayout(semantic_row)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Semantic threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(cfg.semantic.threshold)
        self.threshold_spin.valueChanged.connect(self.save)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        layout.addLayout(threshold_row)

        embedder_row = QHBoxLayout()
        embedder_row.addWidget(QLabel("Embedder:"))
        self.embedder_combo = QComboBox()
        self.embedder_combo.addItem("Hashing (built-in, no dependencies)", "hashing")
        self.embedder_combo.addItem(
            "sentence-transformers (optional install)", "sentence_transformers"
        )
        index = self.embedder_combo.findData(cfg.semantic.embedder)
        if index >= 0:
            self.embedder_combo.setCurrentIndex(index)
        self.embedder_combo.currentIndexChanged.connect(self.save)
        embedder_row.addWidget(self.embedder_combo)
        embedder_row.addStretch()
        layout.addLayout(embedder_row)

        layout.addWidget(
            QLabel(
                "Semantic matching compares only entries with the same context, "
                "system prompt, parameters, and model scope. Numbers, negations, "
                "and prompt length must also agree."
            )
        )

        self.stats_label = QLabel("No cache activity recorded yet.")
        layout.addWidget(self.stats_label)

        btn_row = QHBoxLayout()
        refresh_stats = QPushButton("Refresh stats")
        refresh_stats.clicked.connect(self.refresh)
        btn_row.addWidget(refresh_stats)
        self.clear_btn = QPushButton("Clear cache")
        self.clear_btn.clicked.connect(self.clear)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        self.refresh()

    def save(self) -> None:
        """Persist the current control values to cache.yaml and refresh the engine."""
        cfg = load_cache_config()
        cfg.enabled = self.enabled_cb.isChecked()
        cfg.ttl_seconds = self.ttl_spin.value() * 3600
        cfg.max_entries = self.max_entries_spin.value()
        cfg.semantic = SemanticCacheConfig(
            enabled=self.semantic_cb.isChecked(),
            threshold=self.threshold_spin.value(),
            embedder=self.embedder_combo.currentData() or "hashing",
            model_name=cfg.semantic.model_name,
            hashing_dim=cfg.semantic.hashing_dim,
            max_candidates=cfg.semantic.max_candidates,
        )
        save_cache_config(cfg)
        if self.engine is not None:
            self.engine.reload_cache_config()
        self.refresh()

    def refresh(self) -> None:
        """Reload cache metrics from the response-cache table and update the label."""
        if self.storage is None:
            self.stats_label.setText("No cache activity recorded yet.")
            return
        metrics = self.storage.cache_metrics()
        self.stats_label.setText(
            "Turns: {total} (hits {hits}: exact {exact}, semantic {semantic}; "
            "misses {misses} · hit rate {rate:.0f}%) · Saved: ${saved:.4f}".format(
                total=metrics["total_cacheable_turns"],
                hits=metrics["hits"],
                exact=metrics["exact_hits"],
                semantic=metrics["semantic_hits"],
                misses=metrics["misses"],
                rate=metrics["hit_rate"] * 100,
                saved=metrics["saved_cost_usd"],
            )
        )

    def clear(self) -> None:
        """Delete all cached answers after confirmation."""
        if self.storage is None:
            return
        answer = QMessageBox.question(
            self,
            "ModelMesh",
            "Delete all cached answers?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.storage.clear_response_cache()
            self.refresh()
