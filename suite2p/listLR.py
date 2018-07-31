### custom QPushButton class that plots image when clicked
# requires buttons to put into a QButtonGroup (parent.viewbtns)
# allows up to 1 button to pressed at a time
class ViewButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(ViewButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        ischecked  = parent.viewbtns.checkedId()
        waschecked = parent.btnstate[bid]
        for n in range(len(parent.btnstate)):
            parent.btnstate[n] = False
        if ischecked==bid and not waschecked:
            parent.viewbtns.setExclusive(True)
            parent.ops_plot[1] = bid+1
            M = fig.draw_masks(parent)
            parent.plot_masks(M)
            parent.btnstate[bid]=True
        elif ischecked==bid and waschecked:
            parent.viewbtns.setExclusive(False)
            parent.btnstate[bid]=False
            parent.ops_plot[1] = 0
            M = fig.draw_masks(parent)
            parent.plot_masks(M)
        self.setChecked(parent.btnstate[bid])
        
### custom QDialog which makes a list of items you can include/exclude
class ListChooser(QtGui.QDialog):
    def __init__(self, Text, parent=None):
        super(ListChooser, self).__init__(parent)
        self.setGeometry(300,300,650,320)
        self.setWindowTitle(Text)
        self.win = QtGui.QWidget(self)
        layout = QtGui.QGridLayout()
        self.win.setLayout(layout)
        #self.setCentralWidget(self.win)
        loadtext = QtGui.QPushButton('Load txt file')
        loadtext.resize(loadtext.minimumSizeHint())
        loadtext.clicked.connect(self.load_text)
        layout.addWidget(loadtext,0,0,1,1)
        self.leftlist = QtGui.QListWidget(parent)
        self.rightlist = QtGui.QListWidget(parent)
        layout.addWidget(QtGui.QLabel('INCLUDE'),1,0,1,1)
        layout.addWidget(QtGui.QLabel('EXCLUDE'),1,3,1,1)
        layout.addWidget(self.leftlist,2,0,5,1)
        #self.leftlist.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        sright = QtGui.QPushButton('-->')
        sright.resize(sright.minimumSizeHint())
        sleft = QtGui.QPushButton('<--')
        sleft.resize(sleft.minimumSizeHint())
        sright.clicked.connect(self.move_right)
        sleft.clicked.connect(self.move_left)
        layout.addWidget(sright,3,1,1,1)
        layout.addWidget(sleft,4,1,1,1)
        layout.addWidget(self.rightlist,2,3,5,1)
        done = QtGui.QPushButton('OK')
        done.resize(done.minimumSizeHint())
        done.clicked.connect(lambda: self.exit_list(parent))
        layout.addWidget(done,7,1,1,1)
    def move_right(self):
        currentRow = self.leftlist.currentRow()
        if self.leftlist.item(currentRow) is not None:
            self.rightlist.addItem(self.leftlist.item(currentRow).text())
            self.leftlist.takeItem(currentRow)
    def move_left(self):
        currentRow = self.rightlist.currentRow()
        if self.rightlist.item(currentRow) is not None:
            self.leftlist.addItem(self.rightlist.item(currentRow).text())
            self.rightlist.takeItem(currentRow)
    def load_text(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if name:
            try:
                txtfile = open(name[0], 'r')
                files = txtfile.read()
                txtfile.close()
                files = files.splitlines()
                for f in files:
                    self.leftlist.addItem(f)
            except (OSError, RuntimeError, TypeError, NameError):
                print('not a good list')
    def exit_list(self, parent):
        parent.trainfiles = []
        for n in range(len(self.leftlist)):
            if self.leftlist.item(n) is not None:
                parent.trainfiles.append(self.leftlist.item(n).text())
        self.accept()
