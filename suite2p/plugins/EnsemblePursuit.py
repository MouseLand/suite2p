from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from scipy.stats import zscore
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
from sklearn.decomposition import PCA
import time
import sys,os
from EnsemblePursuit.EnsemblePursuit import EnsemblePursuit
from PyQt5.QtWidgets import QGridLayout, QWidget, QLabel, QPushButton, QComboBox
import matplotlib.pyplot as plt
from pylab import *
from ..gui import menus, io, merge, views, buttons, classgui, traces, graphics, masks
from pyqtgraph.Qt import QtCore
from scipy.stats.stats import pearsonr
try:
    from EnsemblePursuit.EnsemblePursuit import EnsemblePursuit
    ENSEMBLEPURSUIT = True
except:
    ENSEMBLEPURSUIT = False




class ArrowWidgt(QtGui.QWidget):
    '''
    Custom widget for flipping through ensembles that a neuron belongs to.
    The widget is situated in the upper right corner of the EP window.
    You can type the index of the neuron in the box in the middle.
    Attributes
    -------------
    ep_win: the current ensemble pursuit window.
    parent: the main window.
    current_ens_text: text indicating the current selected ensemble that the indicated neuron belongs to.
    neuron: the indicated neuron which can belong to multiple ensembles.
    ens_ind: the index of the currently indicated ensemble.
    ens_lst: all the ensembles that the indicated neuron belongs to.
    '''
    def __init__(self,ep_win,parent=None):
        super(ArrowWidgt, self).__init__(parent)

        self.ep_win=ep_win
        self.parent=parent

        button_left = QtGui.QToolButton()
        button_left.setArrowType(QtCore.Qt.LeftArrow)
        button_left.setStyleSheet("color: gray;")

        button_right = QtGui.QToolButton()
        button_right.setArrowType(QtCore.Qt.RightArrow)
        button_right.setStyleSheet("color: gray;")

        self.current_ens_text = QtGui.QLabel("Ensemble:"+"0")
        self.current_ens_text.setStyleSheet("color: white;")
        self.current_ens_text.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.current_ens_text.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.neuron=QtGui.QLineEdit(self)
        self.neuron.setText("0")
        self.neuron.setFixedWidth(35)
        current_neuron_txt = QtGui.QLabel("Neuron:")
        current_neuron_txt.setStyleSheet("color: white;")
        current_neuron_txt.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        current_neuron_txt.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        self.compute_ens_lst(int(self.neuron.text()))

        self.ens_ind=0

        lay = QtGui.QHBoxLayout(self)

        for btn in (button_left, self.current_ens_text,current_neuron_txt,self.neuron, button_right):
            lay.addWidget(btn)
            if btn==button_left:
                btn.clicked.connect(lambda: self.select_prev_ens())
            if btn==button_right:
                btn.clicked.connect(lambda: self.select_next_ens())
            if btn==self.neuron:
                self.neuron.returnPressed.connect(lambda: self.new_ensembles())

    def new_ensembles(self):
        '''
        This function computes the ensembles that a neuron belongs to and sets the text of the
        widget accordingly. It selects the cells in the main window from the first ensemble in line.
        '''
        #self.neuron.setText(self.neuron.text())
        self.compute_ens_lst(int(self.neuron.text()))
        self.current_ens_text.setText("Ensemble: "+str(self.ens_lst[0]))
        ensemble_ind=self.ens_lst[0]
        self.select_cells(ensemble_ind)


    def select_prev_ens(self):
        '''
        Function for the back arrow in the widget.
        Flips to the previous ensemble in the ensemble list that the indicated neuron belongs to.
        '''
        #self.compute_ens_lst(int(self.neuron.text()))
        if self.ens_ind>=1:
            self.ens_ind-=1
            self.current_ens_text.setText("Ensemble: "+str(self.ens_lst[self.ens_ind]))
            ensemble_ind=self.ens_lst[self.ens_ind]
            self.select_cells(ensemble_ind)
        elif self.ens_ind==0:
            self.ens_ind=0
            self.current_ens_text.setText("Ensemble: "+str(self.ens_lst[self.ens_ind]))
            ensemble_ind=self.ens_lst[self.ens_ind]
            self.select_cells(ensemble_ind)


    def select_next_ens(self):
        '''
        Function for the back arrow in the widget.
        Flips to the next ensemble in the ensemble list that the indicated neuron belongs to.
        '''
        #self.compute_ens_lst(int(self.neuron.text()))
        if self.ens_ind<len(self.ens_lst)-1:
            self.ens_ind+=1
            self.current_ens_text.setText("Ensemble: "+str(self.ens_lst[self.ens_ind]))
            ensemble_ind=self.ens_lst[self.ens_ind]
            self.select_cells(ensemble_ind)

    def compute_ens_lst(self,neuron_ind):
        '''
        Function for computing which ensembles the indicated neuron belongs to.
        Parameters
        -----------
        neuron_ind: int, the index of the indicated neuron.
        '''
        try:
            self.ens_lst=np.where(self.ep_win.U[neuron_ind,:]>0)[0].flatten().tolist()
        except Exception as e:
            self.ens_lst=[]

    def select_cells(self,ensemble_ind):
        '''
        Selects cells that belong to the ensemble in the main window.
        Parameters
        -----------
        ensemble_ind: int, from which ensemble to select neurons in the main window.
        '''
        self.selected=np.nonzero(self.ep_win.U[:,ensemble_ind])[0]
        self.parent.imerge = []
        if self.selected.size < 5000:
            for n in self.selected:
                self.parent.imerge.append(n)
            self.parent.ichosen = self.parent.imerge[0]
            self.parent.update_plot()
        else:
            print('too many cells selected')

class RangeWidget(QtGui.QWidget):
    '''
    A widget for flipping through scatter plots at the bottom of the window.
    There are 12 plots that fit into one page of scatter plots.
    '''
    def __init__(self,ep_win):
        super(RangeWidget, self).__init__()

        self.ep_win=ep_win

        button_left = QtGui.QToolButton()
        button_left.setArrowType(QtCore.Qt.LeftArrow)
        button_left.setStyleSheet("color: gray;")

        button_right = QtGui.QToolButton()
        button_right.setArrowType(QtCore.Qt.RightArrow)
        button_right.setStyleSheet("color: gray;")

        self.current_range = QtGui.QLabel("Range:"+"0-11")
        self.current_range.setStyleSheet("color: white;")
        self.current_range.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.current_range.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


        lay = QtGui.QHBoxLayout(self)

        for btn in (button_left, self.current_range, button_right):
            lay.addWidget(btn)
            if btn==button_left:
                btn.clicked.connect(lambda: self.go_left())
            if btn==button_right:
                btn.clicked.connect(lambda: self.go_right())

    def go_left(self):
        '''
        Flip between pages of scatter plots.
        '''
        if self.ep_win.ptr-1>=0:
            self.ep_win.switch_between_ensembles(self.parent,'left')
            self.current_range.setText("Range: "+str(self.ep_win.ptr*12)+'-'+str((self.ep_win.ptr+1)*12-1))
        else:
            pass

    def go_right(self):
        '''
        Flip between pages of scatter plots.
        '''
        if ((self.ep_win.ptr+1)*12)<=self.ep_win.n_components:
            self.ep_win.switch_between_ensembles(self.parent,'right')
            if self.ep_win.n_components//12<=self.ep_win.ptr:
                rem=self.ep_win.n_components%12
                self.current_range.setText("Range: "+str(self.ep_win.ptr*12)+'-'+str(self.ep_win.ptr*12+rem-1))
            else:
                self.current_range.setText("Range: "+str(self.ep_win.ptr*12)+'-'+str((self.ep_win.ptr+1)*12-1))
        else:
            pass

class BoxRangeWidget(QtGui.QWidget):
    '''
    A widget for flipping through cluster boxes at the top half of the window.
    There are 70 boxes that fit into one page of boxes.
    '''

    def __init__(self,ep_win,parent):
        super(BoxRangeWidget, self).__init__()

        self.ep_win=ep_win
        self.parent=parent

        button_left = QtGui.QToolButton()
        button_left.setArrowType(QtCore.Qt.LeftArrow)
        button_left.setStyleSheet("color: gray;")

        button_right = QtGui.QToolButton()
        button_right.setArrowType(QtCore.Qt.RightArrow)
        button_right.setStyleSheet("color: gray;")

        self.current_range = QtGui.QLabel("Range:"+"0-69")
        self.current_range.setStyleSheet("color: white;")
        self.current_range.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.current_range.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


        lay = QtGui.QHBoxLayout(self)

        for btn in (button_left, self.current_range, button_right):
            lay.addWidget(btn)
            if btn==button_left:
                btn.clicked.connect(lambda: self.go_left())
            if btn==button_right:
                btn.clicked.connect(lambda: self.go_right())

    def go_left(self):
        '''
        Flip between pages of cluster selection boxes.
        '''
        if self.ep_win.box_block_ind-1>=0:
            self.ep_win.switch_between_box_plots('left',self.ep_win.box_block_ind)
            self.current_range.setText("Range: "+str(self.ep_win.box_block_ind*70)+'-'+str((self.ep_win.box_block_ind+1)*70-1))
        else:
            pass

    def go_right(self):
        '''
        Flip between pages of cluster selection boxes.
        '''
        if ((self.ep_win.box_block_ind+1)*70)<=self.ep_win.n_components:
            self.ep_win.switch_between_box_plots('right',self.ep_win.box_block_ind)
            if self.ep_win.n_components//70<=self.ep_win.box_block_ind:
                rem=self.ep_win.n_components%70
                self.current_range.setText("Range: "+str(self.ep_win.box_block_ind*70)+'-'+str(self.ep_win.box_block_ind*70+rem-1))
            else:
                self.current_range.setText("Range: "+str(self.ep_win.box_block_ind*70)+'-'+str((self.ep_win.box_block_ind+1)*70-1))
        else:
            pass



class HeatMapBox(QLabel):
    '''
    Class for a single clickable box in a heatmap boxplot layout.
    Attributes
    -----------
    ep_win: the ensemble pursuit window.
    ensemble: which ensemble the box represents.
    color_ind: index of the color map of the box.
    '''
    def __init__(self,ep_win,ensemble,color_ind):
        super(HeatMapBox, self).__init__()
        self.setAutoFillBackground(True)
        self.ensemble=ensemble
        self.ep_win=ep_win
        self.color_ind=color_ind
        #self.V_plot=V_plot
        palette = self.palette()
        colormap = cm.get_cmap("viridis")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        color=matplotlib.colors.rgb2hex(lut[color_ind])
        self.qcolor=QtGui.QColor(color)
        palette.setColor(QtGui.QPalette.Window, self.qcolor)

        self.setPalette(palette)



    def mousePressEvent(self, event):
        '''
        An event for selecting a box and passing the selected ensemble index to
        the EP window class. Passing the ensemble to the EP window class is
        necessary for the select_cells function in the main window to work.
        Parameters
        -----------
        event: mouse press event on the ensemble box.
        '''
        for box in self.ep_win.widget_lst:
            palette = box.palette()
            box.qcolor.setAlpha(255)
            palette.setColor(QtGui.QPalette.Window, box.qcolor)

            box.setPalette(palette)

        palette = self.palette()
        self.qcolor.setAlpha(0)
        palette.setColor(QtGui.QPalette.Window, self.qcolor)

        self.setPalette(palette)

        self.ep_win.ensemble=self.ensemble


class ScatterPlot(pg.PlotItem):
    '''
    Class for single scatter plot in the layout of scatter plots.
    '''
    def __init__(self):
        super().__init__()

    def mouseDoubleClickEvent(self,e):
        self.autoRange()

class BoxLayout(pg.LayoutWidget):
    '''
    Custom box layout for box buttons. Supports clearing the boxes when flipping
    through multiple pages.
    '''
    def __init__(self, ep_win):
        super(BoxLayout, self).__init__()
        self.layout = QtGui.QGridLayout()
        ep_win.l0.addWidget(self.layout,2,1,6,12)

    def clear(self):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().close()



class EPWindow(QtGui.QMainWindow):
    '''
    Main EnsemblePursuit window.
    Parameters
    -----------
    parent: the main window of suite2p.
    Attributes
    -----------
    cwidget: central widget of the window.
    l0: grid layout for the window.
    win: window of ensemble pursuit.
    sc_plots: layout for the scatter plots at the bottom of the window.
    box_layout: layout for boxes representing ensembles.
    sp: calcium signals from the main window.
    n_components: number of ensembles to fit.
    ptr: int, index which indicates which page of scatter plots is currently visible.
    box_block_ind: int, index which indicates which page of ensemble heatmap boxes is currently visible.
    box_rng: range for box plot.
    box_rm: ensemble boxes that don't fill a single line in the heatmap plot.
    range: scatter plot range widget.
    epOn: button to compute ensemble pursuit.
    nr_ens_text: text demarcating the number of ensembles to fit.
    n_ens: text box for inputting how many ensembles to fit.
    lam_text: lambda parameter descriptive text label.
    lam_input: text box for setting the value of the parameter lambda.
    box_plot_arrows: arrows custom widget for flipping between box plot pages.
    arrows: widget for flipping through which ensembles a neuron belongs to.
    ens_selector: button for highlighting the cells in the selected ensemble in the main window.
    pc: principal components from PCA.
    data_selector: dropdown for selecting which data to plot against ensembles in the scatter plots.
    enter_ind: text box for inputting which data index to plot against the ensembles.
    widget_lst: list, holds the ensemble heatmap boxes.
    color_lst: list, colors for the ensemble heatmap boxes.
    '''
    def __init__(self, parent=None):
        super(EPWindow, self).__init__(parent)
        print('EP Window')
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1500,1500)
        self.setWindowTitle('Visualize data')
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,0,0,14,14)
        layout = self.win.ci.layout

        #Layout widget for plot_scatter
        self.sc_plots=pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.sc_plots,9,0,14,14)
        #self.l0.addItem(self.sc_plots)

        self.make_top(parent)

        self.make_data_selector(parent)


        self.sp=parent.Fbin.copy().T

        self.type_of_plot='boxes'

        if self.type_of_plot=='boxes':
            self.box_layout = QGridLayout()
            box_widget = QWidget()
            box_widget.setLayout(self.box_layout)
            self.l0.addWidget(box_widget,2,1,6,12)


        nt,nn=self.sp.shape

        self.compute_PCA()

        self.ptr=0

        self.box_block_ind=0

        self.box_rng=np.arange(self.box_block_ind*70,(self.box_block_ind+1)*70)
        self.box_rem=0
        if self.n_components//70<=self.box_block_ind:
            self.box_rem=self.n_components%70
            self.box_rng=np.arange((self.box_block_ind)*70,(self.box_block_ind)*70+self.box_rem)



    def make_top(self,parent):
        '''
        Function that collects together making widgets for the very top of the EP window.
        '''
        if not ENSEMBLEPURSUIT:
            self.install_EP_text=QtGui.QLabel("If EnsemblePursuit is not installed, run \"pip install EnsemblePursuit\".")
            self.install_EP_text.setStyleSheet("color: white;")
            self.install_EP_text.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
            self.l0.addWidget(self.install_EP_text, 0,0,1,5)

        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")

        self.epOn = QtGui.QPushButton('compute EnsemblePursuit')
        if not ENSEMBLEPURSUIT:
            self.epOn.setStyleSheet(self.styleInactive)
        else:
            self.epOn.setStyleSheet(self.styleUnpressed)
        self.epOn.clicked.connect(lambda: self.compute_ep(parent))
        self.l0.addWidget(self.epOn,1,0,1,2)

        self.nr_ens_text = QtGui.QLabel("Nr of ensembles=")
        self.nr_ens_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.nr_ens_text.setStyleSheet("color: white;")
        self.nr_ens_text.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.l0.addWidget(self.nr_ens_text,  1, 2.5, 1, 2)

        self.n_ens=QtGui.QLineEdit()
        self.n_ens.setText("25")
        self.n_ens.setFixedWidth(35)
        self.l0.addWidget(self.n_ens, 1, 4.5, 1, 2)
        self.n_ens.returnPressed.connect(lambda: self.set_n_ens())
        self.n_components=int(self.n_ens.text())

        self.lam_text = QtGui.QLabel("Lambda:")
        self.lam_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lam_text.setStyleSheet("color: white;")
        self.lam_text.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.l0.addWidget(self.lam_text,  1, 5.5, 1, 1)

        self.lam_input=QtGui.QLineEdit()
        self.lam_input.setText("0.01")
        self.lam_input.setFixedWidth(35)
        self.lam_input.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignVCenter)
        self.lam_input.returnPressed.connect(lambda: self.set_lambda())
        self.l0.addWidget(self.lam_input, 1, 6.5, 1, 1)
        self.lam=float(self.lam_input.text())


        self.box_plot_arrows=BoxRangeWidget(self,parent)
        self.l0.addWidget(self.box_plot_arrows,1,7.5,1,1)

        self.arrows=ArrowWidgt(self,parent)
        self.l0.addWidget(self.arrows, 1, 11, 1, 2)

        self.ens_selector=QtGui.QPushButton('Select cells')
        self.ens_selector.clicked.connect(lambda: self.select_cells_main(parent))
        self.l0.addWidget(self.ens_selector,1,8.5,1,1)

        self.widget_lst=[]

    def select_cells_main(self,parent):
        '''
        Function that connects to the 'Select Cells' button and plots
        the cells for the ensemble that is selected in the boxes of ensembles subplot
        to display them in the main window of the GUI.
        '''
        self.selected=np.nonzero(self.U[:,self.ensemble])[0]
        parent.imerge = []
        if self.selected.size < 5000:
            for n in self.selected:
                parent.imerge.append(n)
            parent.ichosen = parent.imerge[0]
            parent.update_plot()
        else:
            print('too many cells selected')

    def set_n_ens(self):
        self.n_components=int(self.n_ens.text())

    def set_lambda(self):
        self.lam=float(self.lam_input.text())


    def compute_ep(self,parent):
        '''
        Function that connects to the 'Compute EnsemblePursuit' button and runs EnsemblePursuit.
        '''
        ops = {'n_components': self.n_components, 'lam':self.lam}
        self.error=False
        self.finish=True
        self.epOn.setStyleSheet(self.stylePressed)
        tic=time.time()
        #try:
        model = EnsemblePursuit(n_components=ops['n_components'],lam=ops['lam'],n_kmeans=ops['n_components']).fit(self.sp)
        self.U=model.weights
        non_empty_ens=[]
        ens_with_neurons=np.count_nonzero(self.U,axis=0)
        self.U=self.U[:,ens_with_neurons!=0]
        self.V=model.components_

        self.n_components=self.U.shape[1]

        print('ep computed in %3.2f s'%(time.time()-tic))
                #self.activate(parent)

        self.box_block_ind=0

        self.box_rng=np.arange(self.box_block_ind*70,(self.box_block_ind+1)*70)
        self.box_rem=0
        if self.n_components//70<=self.box_block_ind:
            self.box_rem=self.n_components%70
            self.box_rng=np.arange((self.box_block_ind)*70,(self.box_block_ind)*70+self.box_rem)

        self.plot_boxes(parent)

        self.plot_multiple_scatter(parent,0,np.arange(0,12),'Principal Components',self.ind)

        self.box_plot_arrows.current_range.setText("Range:"+"0-"+str(min(self.n_components-1,69)))


    def set_n_pc(self):
        self.n_pc=int(self.n_pcs.text())

    def make_data_selector(self,parent):
        '''
        The function makes widgets for controlling the type of data that is plotted
        in the scatter plots array at the bottom of the window.
        '''
        self.data_selector=QComboBox()
        data_labels=['Principal Components','Behavior','Individual Neurons','Other Ensembles']
        for j in range(len(data_labels)):
            self.data_selector.addItem(data_labels[j])
        self.data_selector.currentTextChanged.connect(lambda: self.set_data_type(parent))
        self.l0.addWidget(self.data_selector,8,0,1,2)
        self.dat_type=self.data_selector.currentText()

        self.ind=0

        self.enter_ind=QtGui.QLineEdit(self)
        ind_text = QtGui.QLabel("Index=")
        ind_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ind_text.setStyleSheet("color: white;")
        ind_text.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.l0.addWidget(ind_text,  8, 2, 1, 1)
        self.enter_ind.setText("0")
        self.enter_ind.setFixedWidth(35)
        #self.n_ensembles.setStyleSheet("color: white;")
        self.enter_ind.returnPressed.connect(lambda: self.set_ind(parent))
        self.l0.addWidget(self.enter_ind, 8,3, 1, 1)
        self.ind=int(self.enter_ind.text())

        self.range=RangeWidget(self)
        self.l0.addWidget(self.range,8,5,1,1)


    def set_data_type(self,parent):
        '''
        Function for setting the data type for the scatter plots.
        Retrieves value from data_selector widget.
        '''
        self.dat_type=self.data_selector.currentText()
        self.sc_plots.clear()
        self.ptr=0
        self.plot_multiple_scatter(parent,0,np.arange(0,12),self.dat_type,self.ind)
        self.range.current_range.setText("Range:"+"0-11")

    def set_ind(self,parent):
        self.ind=int(self.enter_ind.text())
        self.sc_plots.clear()
        self.ptr=0
        self.plot_multiple_scatter(parent,0,np.arange(0,12),self.dat_type,self.ind)
        self.range.current_range.setText("Range:"+"0-11")


    def set_n_pc(self):
        self.n_pc=int(self.n_pcs.text())

    def set_n_ens(self):
        self.n_components=int(self.n_ens.text())

    def compute_PCA(self):
        pca=PCA(n_components=self.n_components)
        self.pc=pca.fit_transform(self.sp)

    def plot_one_box(self,ens_cntr,nonz,vertical_ind,horizontal_ind):
        '''
        Plots one box in the ensemble box array at the top of the window.
        '''
        #def __init__(self,ep_win,ensemble,color_ind):
        hbox=HeatMapBox(self,ens_cntr,self.color_lst[ens_cntr])
        #w_1.setText(str(j)+str(i)+'__'+str(nonz))
        newfont = QtGui.QFont("Arial", 20, QtGui.QFont.Bold)
        hbox.setFont(newfont)
        hbox.setText(str(nonz))
        hbox.setStyleSheet("color: white;")
        hbox.setMaximumHeight(50)
        #hbox.setMaximumWidth(150)
        #self.box_layout.addWidget(w_1,j,i)
        self.widget_lst.append(hbox)
        self.box_layout.addWidget(hbox,vertical_ind,horizontal_ind)

    def compute_global_color_for_heatmap(self):
        nonz_lst=[]
        for ind in range(0,self.n_components):
            nonz=np.nonzero(self.U[:,ind].flatten())[0].shape[0]
            nonz_lst.append(np.log(nonz))
        min_nonz=min(nonz_lst)
        max_nonz=max(nonz_lst)
        self.color_lst=[]
        for ind in range(0,self.n_components):
            color_ind=int(((nonz_lst[ind]-min_nonz)/(max_nonz-min_nonz))*255)
            self.color_lst.append(color_ind)

    def clear_boxes(self):

        for i in reversed(range(self.box_layout.count())):
            self.box_layout.itemAt(i).widget().setParent(None)


    def plot_boxes(self,parent):
        '''
        Plots ensemble boxes at the top of the screen using the plot_one_box function.
        '''
        self.clear_boxes()
        n=self.box_rng.shape[0]//10
        remainder=self.box_rng.shape[0]%10
        ep_win=self
        self.compute_global_color_for_heatmap()
        ens_cntr=0
        aux_U=self.U[:,self.box_rng[:10*n]].reshape((self.sp.shape[1],n,10))
        for j in range(0,n):
            for i in range(0,10):
                nonz=np.nonzero(aux_U[:,j,i].flatten())[0].shape[0]
                self.plot_one_box(self.box_rng[ens_cntr],nonz,j,i)
                ens_cntr+=1
        aux_U=self.U[:,self.box_rng[10*n:10*n+remainder]].reshape((self.sp.shape[1],1,remainder))
        for z in range(0,remainder):
            nonz=np.nonzero(aux_U[:,0,z].flatten())[0].shape[0]
            self.plot_one_box(self.box_rng[ens_cntr],nonz,n,z)
            ens_cntr+=1
        self.win.show()


    def plot_one_sc(self,parent,x_ax_data,cells_inds,dat_type,col_ind,j,i,ens_ind,ind,lut,last_plot=False):
        '''
        Plots one scatter plot in the scatter plot array at the bottom.
        '''
        if dat_type=='Individual Neurons':
            y_ax_data=self.sp.T[ind,:].flatten()
        elif dat_type=='Principal Components':
            y_ax_data=self.pc.T[ind,:].flatten()
        elif dat_type=='Behavior':
            if parent.bloaded==True:
                y_ax_data=parent.beh[ind,:].flatten()
        elif dat_type=='Other Ensembles':
            y_ax_data=self.V[:,ind].flatten()
        if dat_type!='Behavior':
            scatter_plot=ScatterPlot()
            one_sc=self.sc_plots.addItem(scatter_plot)
            color=lut[self.color_lst[cells_inds[col_ind]]]*255
            color[-1]=5
            #print('col',color)
            color=tuple(color)
            if last_plot==True:
                scatter_plot.plot(x_ax_data, y_ax_data, pen=None, symbol='t', symbolPen=None, symbolSize=15, symbolBrush=color)
            else:
                scatter_plot.plot(x_ax_data, y_ax_data, pen=None, symbol='t', symbolPen=None, symbolSize=5, symbolBrush=color)
            scatter_plot.hideAxis('bottom')
            scatter_plot.hideAxis('left')
            r=pearsonr(x_ax_data,y_ax_data)
            corrcoef=r[0]
            scatter_plot.setTitle('# '+str(cells_inds[ens_ind])+', '+'r= %6.4f'%corrcoef)
        elif dat_type=='Behavior' and parent.bloaded==True:
            scatter_plot=ScatterPlot()
            one_sc=self.sc_plots.addItem(scatter_plot)
            color=lut[self.color_lst[cells_inds[col_ind]]]*255
            color[-1]=5
            #print('col',color)
            color=tuple(color)
            if last_plot==True:
                scatter_plot.plot(x_ax_data, y_ax_data, pen=None, symbol='t', symbolPen=None, symbolSize=15, symbolBrush=color)
            else:
                scatter_plot.plot(x_ax_data, y_ax_data, pen=None, symbol='t', symbolPen=None, symbolSize=5, symbolBrush=color)
            scatter_plot.hideAxis('bottom')
            scatter_plot.hideAxis('left')
            r=pearsonr(x_ax_data,y_ax_data)
            corrcoef=r[0]
            scatter_plot.setTitle('# '+str(cells_inds[ens_ind])+', '+'r= %6.4f'%corrcoef)
        elif dat_type=='Behavior' and parent.bloaded==False:
            pass
            print('Behavior not loaded!')

    def plot_multiple_scatter(self,parent,pc_ind,cells_inds,dat_type,ind):
        '''
        Function for plotting multiple scatter plots on one page.
        It uses 'plot_one_sc' function to plot individual scatter plots.
        One page fits 12 individual scatter plots.
        '''
        colormap = cm.get_cmap("viridis")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        #self.file_loader()
        ens_ind=0
        shp=self.V.shape[0]
        if cells_inds.shape[0]==12:
            col_ind=0
            for j in range(0,3):
                self.sc_plots.nextRow()
                for i in range(0,4):
                    self.sc_plots.nextColumn()
                    x_ax_data=self.V[:,cells_inds].reshape(shp,3,4)[:,j,i].flatten()
                    self.plot_one_sc(parent,x_ax_data,cells_inds,dat_type,col_ind,j,i,ens_ind,ind,lut)
                    col_ind+=1
                    ens_ind+=1
        else:
            n=len(cells_inds)//4
            remainder=len(cells_inds)%4
            col_ind=0
            for j in range(0,n):
                self.sc_plots.nextRow()
                for i in range(0,4):
                    self.sc_plots.nextColumn()
                    x_ax_data=self.V[:,cells_inds[:4*n]].reshape(shp,n,4)[:,j,i].flatten()
                    self.plot_one_sc(parent,x_ax_data,cells_inds,dat_type,col_ind,j,i,ens_ind,ind,lut)
                    #self.sc = self.win.addPlot(title="ScatterPlot",row=5, col=0, colspan=15,rowspan=10)
                    col_ind+=1
                    ens_ind+=1
            self.sc_plots.nextRow()
            for i in range(0,remainder):
                    self.sc_plots.nextColumn()
                    x_ax_data=self.V[:,cells_inds[4*n:4*n+remainder]].reshape(shp,1,remainder)[:,0,i].flatten()
                    self.plot_one_sc(parent,x_ax_data,cells_inds,dat_type,col_ind,2,i,ens_ind,ind,lut,last_plot=True)
                    #self.sc = self.win.addPlot(title="ScatterPlot",row=5, col=0, colspan=15,rowspan=10)
                    col_ind+=1
                    ens_ind+=1
        self.win.show()

    def switch_between_box_plots(self,direction,parent):
        '''
        Function for toggling between pages of cell selector boxes at the top.
        '''
        if direction=='right':
            self.box_block_ind+=1
            self.box_rng=np.arange(self.box_block_ind*70,(self.box_block_ind+1)*70)
            if self.n_components//70<=self.box_block_ind:
                rem=self.n_components%70
                self.box_rng=np.arange((self.box_block_ind)*70,(self.box_block_ind)*70+rem)
            self.plot_boxes(parent)
        if direction=='left':
            self.box_block_ind-=1
            self.box_rng=np.arange(self.box_block_ind*70,(self.box_block_ind+1)*70)
            if self.n_components//70<=self.box_block_ind:
                rem=self.n_components%70
                self.box_rng=np.arange((self.box_block_ind-1)*70,(self.box_block_ind+1)*rem)
            self.plot_boxes(parent)

    def switch_between_ensembles(self,parent,direction):
        '''
        Plot for flipping between ensembles vs data scatter plots at the bottom.
        '''
        if direction=='right':
            self.ptr+=1
            rng=np.arange(self.ptr*12,(self.ptr+1)*12)
            if self.n_components//12<=self.ptr:
                rem=self.n_components%12
                rng=np.arange((self.ptr)*12,(self.ptr)*12+rem)
            self.sc_plots.clear()
            self.plot_multiple_scatter(parent,0,rng,self.dat_type,self.ind)
        if direction=='left':
            self.ptr-=1
            rng=np.arange(self.ptr*12,(self.ptr+1)*12)
            if self.n_components//12<=self.ptr:
                rem=self.n_components%12
                rng=np.arange((self.ptr-1)*12,(self.ptr+1)*rem)
            self.sc_plots.clear()
            self.plot_multiple_scatter(parent,0,rng,self.dat_type,self.ind)

    def keyPressEvent(self, e):
        '''
        Left and right key press events for flipping between scatter plots at
        the bottom of the window.
        '''
        if e.key() == QtCore.Qt.Key_Right:
            if (self.ptr*12)<=self.n_components:
                self.switch_between_ensembles(self.parent,'right')
                self.range.current_range.setText("Range: "+str(self.ptr*12)+'-'+str((self.ptr+1)*12-1))
            else:
                pass
        if e.key() == QtCore.Qt.Key_Left:
            if self.ptr>=0:
                self.switch_between_ensembles(self.parent,'left')
                self.range.current_range.setText("Range: "+str(self.ptr*12)+'-'+str((self.ptr+1)*12-1))
            else:
                pass


    def select_cells(self,parent):
        self.selected=np.nonzero(self.U[:,self.selected_ensemble])[0]
        parent.imerge = []
        if self.selected.size < 5000:
            for n in self.selected:
                parent.imerge.append(n)
            parent.ichosen = parent.imerge[0]
            parent.update_plot()
        else:
            print('too many cells selected')
