from suite2p.gui.gui2p import MainWindow


def test_main_window_launches_without_error(qtbot):
    app = MainWindow()
    qtbot.addWidget(app)
