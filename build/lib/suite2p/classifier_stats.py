### pop-up window to choose classifier fields
# not yet added to GUI
self.chooseStat = QtGui.QAction('&Choose stat fields to use in classifier', self)
self.chooseStat.setShortcut('Ctrl+J')
self.chooseStat.triggered.connect(self.choose_stat)
self.chooseStat.setEnabled(False)
self.addAction(self.chooseStat)

    def choose_stat(self):
        swindow = QtGui.QMainWindow(self)
        swindow.show()
        swindow.setGeometry(700,300,350,400)
        swindow.setWindowTitle('stats for classifier')
        win = QtGui.QWidget(swindow)
        #win.setWindowTitle('Image List')
        win.setMinimumSize(300, 400)
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        self.statlist = QtGui.QListWidget(win)
        self.classlist = QtGui.QListWidget(win)
        for key in self.stat[0]:
            try:
                lkey = len(self.stat[0][key])
            except (TypeError):
                lkey = 1
            if lkey == 1:
                self.statlabels.append(key)
                statlist.addItem(key)
        layout.addWidget(statlist,0,0,4,1)
        sright = QtGui.QPushButton('-->')
        sright.resize(sright.minimumSizeHint())
        sright.clicked.connect(self.add_to_class)
        sleft.clicked.connect(self.remove_from_class)
        sleft = QtGui.QPushButton('<--')
        sleft.resize(sleft.minimumSizeHint())
        layout.addWidget(sright,1,1,1,1)
        layout.addWidget(sleft,2,1,1,1)
        layout.addWidget(classlist,0,2,4,1)
        done = QtGui.QPushButton('OK')
        done.resize(done.minimumSizeHint())
        layout.addWidget(done,4,1,1,1)

        win.show()
        self.statlabels = []
        statchosen = True
        if statchosen:
            self.loadTrain.setEnabled(True)
            self.loadText.setEnabled(True)

    def add_to_class(self):
        print(self.statlist.itemClicked)
    def remove_from_class(self):
        print(self.stat)

    def add_item(self):
        print('yo')
