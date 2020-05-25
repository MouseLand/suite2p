import numpy as np
from PyQt5 import QtGui, QtCore


def plot_trace(parent):
    parent.p3.clear()
    ax = parent.p3.getAxis('left')
    if len(parent.imerge)==1:
        n = parent.imerge[0]
        f = parent.Fcell[n,:]
        fneu = parent.Fneu[n,:]
        sp = parent.Spks[n,:]
        fmax = np.maximum(f.max(), fneu.max())
        fmin = np.minimum(f.min(), fneu.min())
        #sp from 0 to fmax
        sp /= sp.max()
        #agus
        sp *= fmax - fmin
        #sp += fmin*0.95
        if parent.tracesOn:
            parent.p3.plot(parent.trange,f,pen='b')
        if parent.neuropilOn:
            parent.p3.plot(parent.trange,fneu,pen='r')
        if parent.deconvOn:
            parent.p3.plot(parent.trange,(sp+fmin),pen=(255,255,255,100))
        parent.fmin= fmin
        parent.fmax=fmax
        ax.setTicks(None)
    else:
        nmax = int(parent.ncedit.text())
        kspace = 1.0/parent.sc
        ttick = list()
        pmerge = parent.imerge[:np.minimum(len(parent.imerge),nmax)]
        k=len(pmerge)-1
        i = parent.activityMode
        favg = np.zeros((parent.Fcell.shape[1],))
        for n in pmerge[::-1]:
            if i==0:
                f = parent.Fcell[n,:]
            elif i==1:
                f = parent.Fneu[n,:]
            elif i==2:
                f = parent.Fcell[n,:] - 0.7*parent.Fneu[n,:]
            else:
                f = parent.Spks[n,:]
            favg += f.flatten()
            fmax = f.max()
            fmin = f.min()
            f = (f - fmin) / (fmax - fmin)
            rgb = parent.colors['cols'][0][n,:]
            parent.p3.plot(parent.trange,f+k*kspace,pen=rgb)
            ttick.append((k*kspace+f.mean(), str(n)))
            k-=1
        bsc = len(pmerge)/25 + 1
        favg -= favg.min()
        favg /= favg.max()
        # at bottom plot behavior and avg trace
        parent.fmin=0
        if len(pmerge)>5:
            parent.p3.plot(parent.trange,-1*bsc+favg*bsc,pen=(140,140,140))
            parent.fmin=-1*bsc
        if parent.bloaded:
            parent.p3.plot(parent.trange,-1*bsc+favg*bsc,pen=(140,140,140))
            parent.p3.plot(parent.beh_time,-1*bsc+parent.beh*bsc,pen='w')
            parent.fmin=-1*bsc
            #parent.traceLabel[0].setText("<font color='gray'>mean activity</font>")
            #parent.traceLabel[1].setText("<font color='white'>1D variable</font>")
            #parent.traceLabel[2].setText("")
            #ck.append((-0.5*bsc,'1D var'))

        parent.fmax=(len(pmerge)-1)*kspace + 1
        ax.setTicks([ttick])
    #parent.p3.setXRange(0,parent.Fcell.shape[1])
    parent.p3.setYRange(parent.fmin,parent.fmax)

def make_buttons(parent, b0):
    # combo box to decide what kind of activity to view
    qlabel = QtGui.QLabel(parent)
    qlabel.setText("<font color='white'>Activity mode:</font>")
    parent.l0.addWidget(qlabel, b0, 0, 1, 1)
    parent.comboBox = QtGui.QComboBox(parent)
    parent.comboBox.setFixedWidth(100)
    parent.l0.addWidget(parent.comboBox, b0+1, 0, 1, 1)
    parent.comboBox.addItem("F")
    parent.comboBox.addItem("Fneu")
    parent.comboBox.addItem("F - 0.7*Fneu")
    parent.comboBox.addItem("deconvolved")
    parent.activityMode = 3
    parent.comboBox.setCurrentIndex(parent.activityMode)
    parent.comboBox.currentIndexChanged.connect(parent.mode_change)

    # up/down arrows to resize view
    parent.level = 1
    parent.arrowButtons = [
        QtGui.QPushButton(u" \u25b2"),
        QtGui.QPushButton(u" \u25bc"),
    ]
    parent.arrowButtons[0].clicked.connect(lambda: expand_trace(parent))
    parent.arrowButtons[1].clicked.connect(lambda: collapse_trace(parent))
    b = 0
    for btn in parent.arrowButtons:
        btn.setMaximumWidth(22)
        btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        btn.setStyleSheet(parent.styleUnpressed)
        parent.l0.addWidget(
            btn,
            b0+b, 1, 1, 1,
            QtCore.Qt.AlignRight
        )
        b += 1

    parent.pmButtons = [QtGui.QPushButton(" +"), QtGui.QPushButton(" -")]
    parent.pmButtons[0].clicked.connect(lambda: expand_scale(parent))
    parent.pmButtons[1].clicked.connect(lambda: collapse_scale(parent))
    b = 0
    parent.sc = 2
    for btn in parent.pmButtons:
        btn.setMaximumWidth(22)
        btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        btn.setStyleSheet(parent.styleUnpressed)
        parent.l0.addWidget(btn, b0 + b, 1, 1, 1)
        b += 1
    # choose max # of cells plotted
    parent.l0.addWidget(
        QtGui.QLabel("<font color='white'>max # plotted:</font>"),
        b0+2,
        0,
        1,
        1,
    )
    b0+=3
    parent.ncedit = QtGui.QLineEdit(parent)
    parent.ncedit.setValidator(QtGui.QIntValidator(0, 400))
    parent.ncedit.setText("40")
    parent.ncedit.setFixedWidth(35)
    parent.ncedit.setAlignment(QtCore.Qt.AlignRight)
    parent.ncedit.returnPressed.connect(lambda: nc_chosen(parent))
    parent.l0.addWidget(parent.ncedit, b0, 0, 1, 1)
    #Agus
    # Deconv CHECKBOX
    parent.l0.setVerticalSpacing(4)
    parent.checkBoxd = QtGui.QCheckBox("deconv [N]")
    parent.checkBoxd.setStyleSheet("color: white;")
    parent.checkBoxd.toggled.connect(lambda: deconv_on(parent))
    parent.deconvOn = True
    parent.checkBoxd.toggle()
    parent.l0.addWidget(parent.checkBoxd,
    b0,
    3,
    1, 2)
    # neuropil CHECKBOX
    parent.l0.setVerticalSpacing(4)
    parent.checkBoxn = QtGui.QCheckBox("neuropil [B]")
    parent.checkBoxn.setStyleSheet("color: red;")
    parent.checkBoxn.toggled.connect(lambda: neuropil_on(parent))
    parent.neuropilOn = True
    parent.checkBoxn.toggle()
    parent.l0.addWidget(parent.checkBoxn,
    b0,5,
    1, 2)
    # traces CHECKBOX
    parent.l0.setVerticalSpacing(4)
    parent.checkBoxt = QtGui.QCheckBox("raw fluor [V]")
    parent.checkBoxt.setStyleSheet("color: blue;")
    parent.checkBoxt.toggled.connect(lambda: traces_on(parent))
    parent.tracesOn = True
    parent.checkBoxt.toggle()
    parent.l0.addWidget(parent.checkBoxt,
    b0,7,
    1, 2)
    return b0

def expand_scale(parent):
    parent.sc += 0.5
    parent.sc = np.minimum(10, parent.sc)
    plot_trace(parent)
    parent.show()

def collapse_scale(parent):
    parent.sc -= 0.5
    parent.sc = np.maximum(0.5, parent.sc)
    plot_trace(parent)
    parent.show()

def expand_trace(parent):
    parent.level += 1
    parent.level = np.minimum(5, parent.level)
    parent.win.ci.layout.setRowStretchFactor(1, parent.level)
    #parent.p1.zoom_plot()

def collapse_trace(parent):
    parent.level -= 1
    parent.level = np.maximum(1, parent.level)
    parent.win.ci.layout.setRowStretchFactor(1, parent.level)
    #parent.p1.zoom_plot()

def nc_chosen(parent):
    if parent.loaded:
        plot_trace(parent)
        parent.show()

#Agus
def deconv_on(parent):
    state = parent.checkBoxd.isChecked()
    if parent.loaded:
        if state:
            parent.deconvOn = True
        else:
            parent.deconvOn = False
        plot_trace(parent)
        parent.win.show()
        parent.show()

def neuropil_on(parent):
    state = parent.checkBoxn.isChecked()
    if parent.loaded:
        if state:
            parent.neuropilOn = True
        else:
            parent.neuropilOn = False
        plot_trace(parent)
        parent.win.show()
        parent.show()

def traces_on(parent):
    state = parent.checkBoxt.isChecked()
    if parent.loaded:
        if state:
            parent.tracesOn = True
        else:
            parent.tracesOn = False
        plot_trace(parent)
        parent.win.show()
        parent.show()
