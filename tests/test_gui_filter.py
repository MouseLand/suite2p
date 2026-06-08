import os
import numpy as np
import pytest
from qtpy import QtWidgets, QtCore, QtGui

@pytest.fixture(scope="module")
def gui():
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])
        
    from suite2p.gui.gui2p import MainWindow
    
    statfile = '/mnt/other_ubunthu/mnt/data/1-ordered/Stav1/stat.npy'
    if not os.path.exists(statfile):
        pytest.skip(f"Test dataset {statfile} not found")
        
    gui_instance = MainWindow(statfile=statfile)
    app.processEvents()
    yield gui_instance
    gui_instance.close()

def test_gui_initial_state(gui):
    # Verify the initial counter matches total ROIs (3289)
    assert "3289 / 3289 ROIs" in gui.filter_counter_label.text()
    assert not gui.filter_checkbox.isChecked()

def test_gui_filter_by_range_all(gui):
    app = QtWidgets.QApplication.instance()
    # Reset filter state first
    gui.filter_class_combo.setCurrentText("All")
    gui.filter_min_prob.setText("0.3")
    gui.filter_max_prob.setText("0.7")
    
    # Enable filter
    gui.filter_checkbox.setChecked(True)
    app.processEvents()
    
    probs = gui.probcell
    expected_matching_all = ((probs >= 0.3) & (probs <= 0.7)).sum()
    assert f"{expected_matching_all} / 3289" in gui.filter_counter_label.text()

def test_gui_filter_by_range_cells(gui):
    app = QtWidgets.QApplication.instance()
    gui.filter_checkbox.setChecked(True)
    gui.filter_min_prob.setText("0.3")
    gui.filter_max_prob.setText("0.7")
    gui.filter_class_combo.setCurrentText("Cells")
    app.processEvents()
    
    probs = gui.probcell
    expected_matching_cells = (((probs >= 0.3) & (probs <= 0.7)) & (gui.iscell == 1)).sum()
    assert f"{expected_matching_cells} / 3289" in gui.filter_counter_label.text()

def test_gui_filter_by_range_noncells(gui):
    app = QtWidgets.QApplication.instance()
    gui.filter_checkbox.setChecked(True)
    gui.filter_min_prob.setText("0.3")
    gui.filter_max_prob.setText("0.7")
    gui.filter_class_combo.setCurrentText("Non-Cells")
    app.processEvents()
    
    probs = gui.probcell
    expected_matching_noncells = (((probs >= 0.3) & (probs <= 0.7)) & (gui.iscell == 0)).sum()
    assert f"{expected_matching_noncells} / 3289" in gui.filter_counter_label.text()

def test_gui_keyboard_navigation(gui):
    app = QtWidgets.QApplication.instance()
    gui.filter_checkbox.setChecked(True)
    gui.filter_class_combo.setCurrentText("Cells")
    gui.filter_min_prob.setText("0.8")
    gui.filter_max_prob.setText("0.95")
    app.processEvents()
    
    matching = gui.get_matching_rois()
    matching_cells = [i for i in range(len(gui.stat)) if gui.iscell[i] == 1 and matching[i]]
    
    # Select the first matching cell
    gui.ichosen = matching_cells[0]
    gui.imerge = [gui.ichosen]
    gui.update_plot()
    app.processEvents()
    
    # Simulate Right arrow key press (Key_Right)
    event_right = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    gui.keyPressEvent(event_right)
    app.processEvents()
    
    # Assert that the new chosen cell matches the filter and is cell
    assert gui.iscell[gui.ichosen] == 1
    assert matching[gui.ichosen]

    # Assert that the classifier prob label displays the correct value
    expected_prob_str = "%2.4f" % (gui.probcell[gui.ichosen])
    assert expected_prob_str in gui.ROIprob.text()

def test_gui_filter_disable(gui):
    app = QtWidgets.QApplication.instance()
    # Disable filter and assert reset
    gui.filter_checkbox.setChecked(False)
    app.processEvents()
    assert "3289 / 3289" in gui.filter_counter_label.text()
