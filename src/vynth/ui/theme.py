"""Professional dark theme for Vynth."""

import sys


class Colors:
    """Color palette for the Vynth dark theme."""

    BG_DARKEST = "#0a0a0f"
    BG_DARK = "#12121a"
    BG_MEDIUM = "#1a1a2e"
    BG_LIGHT = "#252540"
    BG_HIGHLIGHT = "#2d2d50"

    ACCENT_PRIMARY = "#6c63ff"
    ACCENT_SECONDARY = "#00d4aa"
    ACCENT_WARM = "#ff6b6b"
    ACCENT_GOLD = "#ffd93d"

    TEXT_PRIMARY = "#e8e8f0"
    TEXT_SECONDARY = "#8888aa"
    TEXT_DIM = "#555570"

    METER_GREEN = "#00e676"
    METER_YELLOW = "#ffeb3b"
    METER_RED = "#ff1744"

    WAVEFORM = "#6c63ff"
    WAVEFORM_BG = "#0d0d15"
    GRID = "#1a1a30"

    BORDER = "#2a2a45"
    BORDER_LIGHT = "#3a3a60"


def get_font_family() -> str:
    """Return the preferred font family for the current platform."""
    if sys.platform == "win32":
        return "Segoe UI"
    if sys.platform == "darwin":
        return "SF Pro Text"
    return "Noto Sans"


def get_stylesheet() -> str:
    """Return the full QSS stylesheet for the Vynth application."""
    c = Colors
    font = get_font_family()

    return f"""
    /* ── Global ───────────────────────────────────────────── */
    * {{
        font-family: "{font}";
        font-size: 13px;
        color: {c.TEXT_PRIMARY};
        outline: none;
    }}

    /* ── QMainWindow / QWidget ────────────────────────────── */
    QMainWindow {{
        background-color: {c.BG_DARKEST};
    }}
    QWidget {{
        background-color: {c.BG_DARKEST};
    }}

    /* ── QPushButton ──────────────────────────────────────── */
    QPushButton {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 5px;
        padding: 6px 16px;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {c.BG_LIGHT};
        border-color: {c.BORDER_LIGHT};
    }}
    QPushButton:pressed {{
        background-color: {c.ACCENT_PRIMARY};
        border-color: {c.ACCENT_PRIMARY};
    }}
    QPushButton:disabled {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_DIM};
        border-color: {c.BG_MEDIUM};
    }}
    QPushButton:checked {{
        background-color: {c.ACCENT_PRIMARY};
        border-color: {c.ACCENT_PRIMARY};
    }}

    /* ── QLabel ────────────────────────────────────────────── */
    QLabel {{
        background-color: transparent;
        color: {c.TEXT_PRIMARY};
        padding: 1px;
    }}

    /* ── QComboBox ─────────────────────────────────────────── */
    QComboBox {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 5px 10px;
        min-height: 20px;
    }}
    QComboBox:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    QComboBox:on {{
        border-color: {c.ACCENT_PRIMARY};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border-left: 1px solid {c.BORDER};
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {c.TEXT_SECONDARY};
        margin-right: 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        selection-background-color: {c.ACCENT_PRIMARY};
        selection-color: {c.TEXT_PRIMARY};
        padding: 4px;
    }}

    /* ── QSpinBox / QDoubleSpinBox ─────────────────────────── */
    QSpinBox, QDoubleSpinBox {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        min-height: 20px;
    }}
    QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {c.ACCENT_PRIMARY};
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid {c.BORDER};
        border-bottom: 1px solid {c.BORDER};
        border-top-right-radius: 4px;
        background-color: {c.BG_MEDIUM};
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
        background-color: {c.BG_LIGHT};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border-left: 1px solid {c.BORDER};
        border-bottom-right-radius: 4px;
        background-color: {c.BG_MEDIUM};
    }}
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {c.BG_LIGHT};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 5px solid {c.TEXT_SECONDARY};
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {c.TEXT_SECONDARY};
    }}

    /* ── QSlider (horizontal) ─────────────────────────────── */
    QSlider::groove:horizontal {{
        background-color: {c.BG_MEDIUM};
        border: 1px solid {c.BORDER};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background-color: {c.ACCENT_PRIMARY};
        border: none;
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background-color: #7d75ff;
    }}
    QSlider::sub-page:horizontal {{
        background-color: {c.ACCENT_PRIMARY};
        border-radius: 3px;
    }}
    QSlider::add-page:horizontal {{
        background-color: {c.BG_MEDIUM};
        border-radius: 3px;
    }}

    /* ── QSlider (vertical) ───────────────────────────────── */
    QSlider::groove:vertical {{
        background-color: {c.BG_MEDIUM};
        border: 1px solid {c.BORDER};
        width: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:vertical {{
        background-color: {c.ACCENT_PRIMARY};
        border: none;
        width: 16px;
        height: 16px;
        margin: 0 -6px;
        border-radius: 8px;
    }}
    QSlider::handle:vertical:hover {{
        background-color: #7d75ff;
    }}
    QSlider::sub-page:vertical {{
        background-color: {c.BG_MEDIUM};
        border-radius: 3px;
    }}
    QSlider::add-page:vertical {{
        background-color: {c.ACCENT_PRIMARY};
        border-radius: 3px;
    }}

    /* ── QScrollBar (horizontal) ──────────────────────────── */
    QScrollBar:horizontal {{
        background-color: {c.BG_DARKEST};
        height: 10px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {c.BG_LIGHT};
        border-radius: 5px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: {c.BG_HIGHLIGHT};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
    }}

    /* ── QScrollBar (vertical) ────────────────────────────── */
    QScrollBar:vertical {{
        background-color: {c.BG_DARKEST};
        width: 10px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c.BG_LIGHT};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {c.BG_HIGHLIGHT};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    /* ── QTabWidget ───────────────────────────────────────── */
    QTabWidget {{
        background-color: {c.BG_DARKEST};
        border: none;
    }}
    QTabWidget::pane {{
        background-color: {c.BG_DARK};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        top: -1px;
    }}

    /* ── QTabBar ──────────────────────────────────────────── */
    QTabBar {{
        background-color: transparent;
    }}
    QTabBar::tab {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_SECONDARY};
        border: 1px solid {c.BORDER};
        border-bottom: none;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        padding: 7px 18px;
        margin-right: 2px;
    }}
    QTabBar::tab:hover {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_PRIMARY};
    }}
    QTabBar::tab:selected {{
        background-color: {c.BG_MEDIUM};
        color: {c.ACCENT_PRIMARY};
        border-color: {c.BORDER_LIGHT};
    }}

    /* ── QGroupBox ────────────────────────────────────────── */
    QGroupBox {{
        background-color: {c.BG_DARK};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        margin-top: 14px;
        padding: 12px 8px 8px 8px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {c.TEXT_SECONDARY};
    }}

    /* ── QDockWidget ──────────────────────────────────────── */
    QDockWidget {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_PRIMARY};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {c.BG_MEDIUM};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 6px 10px;
        text-align: left;
        font-weight: bold;
    }}
    QDockWidget::close-button, QDockWidget::float-button {{
        background-color: transparent;
        border: none;
        padding: 2px;
    }}
    QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
        background-color: {c.BG_LIGHT};
        border-radius: 3px;
    }}

    /* ── QMenuBar ─────────────────────────────────────────── */
    QMenuBar {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_PRIMARY};
        border-bottom: 1px solid {c.BORDER};
        padding: 2px;
    }}
    QMenuBar::item {{
        background-color: transparent;
        padding: 5px 12px;
        border-radius: 4px;
    }}
    QMenuBar::item:hover {{
        background-color: {c.BG_MEDIUM};
    }}
    QMenuBar::item:selected {{
        background-color: {c.BG_LIGHT};
    }}

    /* ── QMenu ────────────────────────────────────────────── */
    QMenu {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 6px;
    }}
    QMenu::item {{
        padding: 6px 30px 6px 20px;
        border-radius: 4px;
    }}
    QMenu::item:hover {{
        background-color: {c.ACCENT_PRIMARY};
        color: {c.TEXT_PRIMARY};
    }}
    QMenu::item:disabled {{
        color: {c.TEXT_DIM};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {c.BORDER};
        margin: 4px 10px;
    }}
    QMenu::indicator {{
        width: 14px;
        height: 14px;
        margin-left: 4px;
    }}

    /* ── QToolBar ──────────────────────────────────────────── */
    QToolBar {{
        background-color: {c.BG_DARK};
        border-bottom: 1px solid {c.BORDER};
        padding: 4px 8px;
        spacing: 6px;
    }}
    QToolBar::separator {{
        width: 1px;
        background-color: {c.BORDER};
        margin: 4px 6px;
    }}
    QToolBar QToolButton {{
        background-color: transparent;
        color: {c.TEXT_PRIMARY};
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 5px 10px;
    }}
    QToolBar QToolButton:hover {{
        background-color: {c.BG_MEDIUM};
        border-color: {c.BORDER};
    }}
    QToolBar QToolButton:pressed {{
        background-color: {c.ACCENT_PRIMARY};
    }}
    QToolBar QToolButton:checked {{
        background-color: {c.BG_LIGHT};
        border-color: {c.ACCENT_PRIMARY};
    }}

    /* ── QStatusBar ───────────────────────────────────────── */
    QStatusBar {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_SECONDARY};
        border-top: 1px solid {c.BORDER};
        padding: 2px;
        font-size: 12px;
    }}
    QStatusBar::item {{
        border: none;
    }}
    QStatusBar QLabel {{
        color: {c.TEXT_SECONDARY};
        padding: 0 8px;
    }}

    /* ── QLineEdit ────────────────────────────────────────── */
    QLineEdit {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 5px 10px;
        selection-background-color: {c.ACCENT_PRIMARY};
    }}
    QLineEdit:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    QLineEdit:focus {{
        border-color: {c.ACCENT_PRIMARY};
    }}
    QLineEdit:disabled {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_DIM};
    }}

    /* ── QTextEdit ────────────────────────────────────────── */
    QTextEdit {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 6px;
        selection-background-color: {c.ACCENT_PRIMARY};
    }}
    QTextEdit:focus {{
        border-color: {c.ACCENT_PRIMARY};
    }}

    /* ── QListWidget ──────────────────────────────────────── */
    QListWidget {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 4px;
        outline: none;
    }}
    QListWidget::item {{
        padding: 5px 8px;
        border-radius: 3px;
    }}
    QListWidget::item:hover {{
        background-color: {c.BG_MEDIUM};
    }}
    QListWidget::item:selected {{
        background-color: {c.ACCENT_PRIMARY};
        color: {c.TEXT_PRIMARY};
    }}

    /* ── QTreeWidget ──────────────────────────────────────── */
    QTreeWidget, QTreeView {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 4px;
        outline: none;
    }}
    QTreeWidget::item, QTreeView::item {{
        padding: 4px 6px;
        border-radius: 3px;
    }}
    QTreeWidget::item:hover, QTreeView::item:hover {{
        background-color: {c.BG_MEDIUM};
    }}
    QTreeWidget::item:selected, QTreeView::item:selected {{
        background-color: {c.ACCENT_PRIMARY};
        color: {c.TEXT_PRIMARY};
    }}
    QTreeWidget::branch, QTreeView::branch {{
        background-color: transparent;
    }}
    QHeaderView::section {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_SECONDARY};
        border: none;
        border-right: 1px solid {c.BORDER};
        border-bottom: 1px solid {c.BORDER};
        padding: 5px 8px;
        font-weight: bold;
    }}

    /* ── QProgressBar ─────────────────────────────────────── */
    QProgressBar {{
        background-color: {c.BG_MEDIUM};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        text-align: center;
        color: {c.TEXT_PRIMARY};
        min-height: 18px;
        font-size: 11px;
    }}
    QProgressBar::chunk {{
        background-color: {c.ACCENT_PRIMARY};
        border-radius: 3px;
    }}

    /* ── QCheckBox ─────────────────────────────────────────── */
    QCheckBox {{
        background-color: transparent;
        color: {c.TEXT_PRIMARY};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        background-color: {c.BG_MEDIUM};
    }}
    QCheckBox::indicator:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c.ACCENT_PRIMARY};
        border-color: {c.ACCENT_PRIMARY};
    }}
    QCheckBox:disabled {{
        color: {c.TEXT_DIM};
    }}

    /* ── QRadioButton ─────────────────────────────────────── */
    QRadioButton {{
        background-color: transparent;
        color: {c.TEXT_PRIMARY};
        spacing: 8px;
    }}
    QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 1px solid {c.BORDER};
        border-radius: 9px;
        background-color: {c.BG_MEDIUM};
    }}
    QRadioButton::indicator:hover {{
        border-color: {c.BORDER_LIGHT};
    }}
    QRadioButton::indicator:checked {{
        background-color: {c.ACCENT_PRIMARY};
        border-color: {c.ACCENT_PRIMARY};
    }}

    /* ── QToolTip ──────────────────────────────────────────── */
    QToolTip {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER_LIGHT};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 12px;
    }}

    /* ── QSplitter ─────────────────────────────────────────── */
    QSplitter {{
        background-color: transparent;
    }}
    QSplitter::handle {{
        background-color: {c.BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    QSplitter::handle:hover {{
        background-color: {c.ACCENT_PRIMARY};
    }}

    /* ── QDialog ───────────────────────────────────────────── */
    QDialog {{
        background-color: {c.BG_DARK};
    }}

    /* ── QFrame separator ─────────────────────────────────── */
    QFrame[frameShape="4"] {{
        color: {c.BORDER};
        max-height: 1px;
    }}
    QFrame[frameShape="5"] {{
        color: {c.BORDER};
        max-width: 1px;
    }}
    """


def apply_theme(app) -> None:
    """Apply the Vynth dark theme to a QApplication instance.

    Parameters
    ----------
    app:
        The ``QApplication`` (or compatible) instance to theme.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont, QPalette

    # Stylesheet
    app.setStyleSheet(get_stylesheet())

    # Font
    font = QFont(get_font_family(), 10)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    # Palette – keeps native dialogs consistent with the stylesheet
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(Colors.BG_DARKEST))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(Colors.BG_DARK))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(Colors.BG_MEDIUM))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(Colors.BG_MEDIUM))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(Colors.BG_MEDIUM))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(Colors.ACCENT_WARM))
    palette.setColor(QPalette.ColorRole.Link, QColor(Colors.ACCENT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(Colors.ACCENT_PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(Colors.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(Colors.TEXT_DIM))

    # Disabled variants
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(Colors.TEXT_DIM)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(Colors.TEXT_DIM)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(Colors.TEXT_DIM)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(Colors.BG_LIGHT)
    )

    app.setPalette(palette)
    app.setStyle("Fusion")
