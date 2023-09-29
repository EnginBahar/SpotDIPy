import pickle

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
import utils as dipu
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mpl_toolkits.basemap import Basemap
from matplotlib.cm import ScalarMappable
import astropy.constants as ac
import astropy.units as au

groupBoxStyle = '''
QGroupBox {
    font-weight: bold;
    background-color: white;
    border: 1px solid green;
    border-radius: 2px;
    margin-top: 20px;
           }

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 5px;
                  }
'''

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, tight_layout=False, projection=None):
        fig = Figure(tight_layout=tight_layout)
        self.axes = fig.add_subplot(111, projection=projection)
        self.fig = fig
        super(MplCanvas, self).__init__(fig)


class BaseCanvas(FigureCanvasQTAgg):
    def __init__(self, incl, tight_layout=True, parent=None):
        self.fig = Figure()

        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        ax4 = plt.subplot2grid((2, 2), (1, 1))

        self.fig.add_subplot(ax1.get_subplotspec())
        self.fig.add_subplot(ax2.get_subplotspec())
        self.fig.add_subplot(ax3.get_subplotspec())
        self.fig.add_subplot(ax4.get_subplotspec())

        super(BaseCanvas, self).__init__(self.fig)

        self.m1 = Basemap(ax=self.fig.axes[0], projection='ortho', lat_0=90 - incl, lon_0=0.0)
        self.m2 = Basemap(ax=self.fig.axes[1], projection='ortho', lat_0=90 - incl, lon_0=-90)
        self.m3 = Basemap(ax=self.fig.axes[2], projection='ortho', lat_0=90 - incl, lon_0=-180)
        self.m4 = Basemap(ax=self.fig.axes[3], projection='ortho', lat_0=90 - incl, lon_0=-270)


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     show_label=False), resizable=True)

class PlotGUI(QtWidgets.QMainWindow):

    def __init__(self, DIP, plot_params, save_maps=None):

        super().__init__()

        self.DIP = DIP
        self.save_maps = save_maps

        self.recons_fss = DIP.recons_result['opt_result'].x.copy()
        self.nit = DIP.recons_result['opt_result'].nit
        self.nfev = DIP.recons_result['opt_result'].nfev
        self.tse = DIP.recons_result['total_spotted_area']
        self.pse = DIP.recons_result['partial_spotted_area']
        self.lmbds = DIP.recons_result['lmbds']
        self.chisqs = DIP.recons_result['chisqs']
        self.mems = DIP.recons_result['mems']
        self.maxcurve = DIP.recons_result['maxcurve']

        self.surface_grid = DIP.surface_grid.copy()

        self.line_sp = plot_params['line_sep_prf']
        self.line_sr = plot_params['line_sep_res']
        self.mol_sp = plot_params['mol_sep_prf']
        self.mol_sr = plot_params['mol_sep_res']
        self.seb = plot_params['show_err_bars']

        self.ms = plot_params['markersize']
        self.fs = plot_params['fontsize']
        self.lw = plot_params['linewidth']
        self.tls = plot_params['ticklabelsize']

        self.cphases = DIP.phases.copy()
        self.pvf_cp = dipu.calc_fs_variation(phases=self.cphases, fss=self.recons_fss,
                                             areas=self.surface_grid['grid_areas'],
                                             lats=self.surface_grid['grid_lats'],
                                             longs=self.surface_grid['grid_longs'], incl=DIP.params['incl'])

        self.aphases = np.linspace(0, 1.0, 1000).copy()
        self.pvf_ap = dipu.calc_fs_variation(phases=self.aphases, fss=self.recons_fss,
                                             areas=self.surface_grid['grid_areas'],
                                             lats=self.surface_grid['grid_lats'],
                                             longs=self.surface_grid['grid_longs'], incl=DIP.params['incl'])

        self.setWindowTitle('Plot and Result Window')
        self.resize(1024, 768)

        main_tab = QtWidgets.QTabWidget(self)

        self.setCentralWidget(main_tab)

        profile_tab = QtWidgets.QTabWidget(self)

        line_profile_frame = QtWidgets.QFrame()
        line_profile_frame_glayout = QtWidgets.QGridLayout(line_profile_frame)

        line_profile_gbox = QtWidgets.QGroupBox('Line Profiles')
        line_profile_gbox_glayout = QtWidgets.QGridLayout(line_profile_gbox)
        line_profile_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        line_profile_gbox_glayout.setSpacing(4)
        line_profile_gbox.setStyleSheet(groupBoxStyle)

        line_residual_gbox = QtWidgets.QGroupBox('Residuals')
        line_residual_gbox_glayout = QtWidgets.QGridLayout(line_residual_gbox)
        line_residual_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        line_residual_gbox_glayout.setSpacing(4)
        line_residual_gbox.setStyleSheet(groupBoxStyle)

        line_sep_profiles_label = QtWidgets.QLabel('Profile Seperation :')
        line_sep_redisuals_label = QtWidgets.QLabel('Residual Seperation :')

        self.line_sep_profiles_ledit = QtWidgets.QLineEdit(str(self.line_sp))
        self.line_sep_residuals_ledit = QtWidgets.QLineEdit(str(self.line_sr))

        self.line_error_bar_cbox = QtWidgets.QCheckBox('Show Errorbar')
        self.line_error_bar_cbox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.line_error_bar_cbox.setChecked(True if self.seb else False)

        line_replot_profiles_button = QtWidgets.QPushButton('Replot')

        line_profile_settings_hlayout = QtWidgets.QHBoxLayout()
        line_profile_settings_hlayout.addStretch()
        line_profile_settings_hlayout.addWidget(line_sep_profiles_label)
        line_profile_settings_hlayout.addWidget(self.line_sep_profiles_ledit)
        line_profile_settings_hlayout.addWidget(line_sep_redisuals_label)
        line_profile_settings_hlayout.addWidget(self.line_sep_residuals_ledit)
        line_profile_settings_hlayout.addWidget(self.line_error_bar_cbox)
        line_profile_settings_hlayout.addWidget(line_replot_profiles_button)
        line_profile_settings_hlayout.addStretch()

        line_profile_frame_glayout.addWidget(line_profile_gbox, 0, 0, 1, 1)
        line_profile_frame_glayout.addWidget(line_residual_gbox, 0, 1, 1, 1)
        line_profile_frame_glayout.addLayout(line_profile_settings_hlayout, 1, 0, 1, 2)

        self.line_profile_plot = MplCanvas(tight_layout=True)
        line_profile_toolbar = NavigationToolbar2QT(self.line_profile_plot, self)

        line_profile_gbox_glayout.addWidget(line_profile_toolbar)
        line_profile_gbox_glayout.addWidget(self.line_profile_plot)

        self.line_residual_plot = MplCanvas(tight_layout=True)
        line_residual_toolbar = NavigationToolbar2QT(self.line_residual_plot, self)

        line_residual_gbox_glayout.addWidget(line_residual_toolbar)
        line_residual_gbox_glayout.addWidget(self.line_residual_plot)

        """"""" mol 1 """""""
        
        mol1_profile_frame = QtWidgets.QFrame()
        mol1_profile_frame_glayout = QtWidgets.QGridLayout(mol1_profile_frame)

        mol1_profile_gbox = QtWidgets.QGroupBox('Molecular Profiles - 1')
        mol1_profile_gbox_glayout = QtWidgets.QGridLayout(mol1_profile_gbox)
        mol1_profile_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        mol1_profile_gbox_glayout.setSpacing(4)
        mol1_profile_gbox.setStyleSheet(groupBoxStyle)

        mol1_residual_gbox = QtWidgets.QGroupBox('Residuals')
        mol1_residual_gbox_glayout = QtWidgets.QGridLayout(mol1_residual_gbox)
        mol1_residual_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        mol1_residual_gbox_glayout.setSpacing(4)
        mol1_residual_gbox.setStyleSheet(groupBoxStyle)

        mol1_sep_profiles_label = QtWidgets.QLabel('Profile Seperation :')
        mol1_sep_redisuals_label = QtWidgets.QLabel('Residual Seperation :')

        self.mol1_sep_profiles_ledit = QtWidgets.QLineEdit(str(self.mol_sp))
        self.mol1_sep_residuals_ledit = QtWidgets.QLineEdit(str(self.mol_sr))

        self.mol1_error_bar_cbox = QtWidgets.QCheckBox('Show Errorbar')
        self.mol1_error_bar_cbox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.mol1_error_bar_cbox.setChecked(True if self.seb else False)

        mol1_replot_profiles_button = QtWidgets.QPushButton('Replot')

        mol1_profile_settings_hlayout = QtWidgets.QHBoxLayout()
        mol1_profile_settings_hlayout.addStretch()
        mol1_profile_settings_hlayout.addWidget(mol1_sep_profiles_label)
        mol1_profile_settings_hlayout.addWidget(self.mol1_sep_profiles_ledit)
        mol1_profile_settings_hlayout.addWidget(mol1_sep_redisuals_label)
        mol1_profile_settings_hlayout.addWidget(self.mol1_sep_residuals_ledit)
        mol1_profile_settings_hlayout.addWidget(self.mol1_error_bar_cbox)
        mol1_profile_settings_hlayout.addWidget(mol1_replot_profiles_button)
        mol1_profile_settings_hlayout.addStretch()

        mol1_profile_frame_glayout.addWidget(mol1_profile_gbox, 0, 0, 1, 1)
        mol1_profile_frame_glayout.addWidget(mol1_residual_gbox, 0, 1, 1, 1)
        mol1_profile_frame_glayout.addLayout(mol1_profile_settings_hlayout, 1, 0, 1, 2)

        self.mol1_profile_plot = MplCanvas(tight_layout=True)
        mol1_profile_toolbar = NavigationToolbar2QT(self.mol1_profile_plot, self)

        mol1_profile_gbox_glayout.addWidget(mol1_profile_toolbar)
        mol1_profile_gbox_glayout.addWidget(self.mol1_profile_plot)

        self.mol1_residual_plot = MplCanvas(tight_layout=True)
        mol1_residual_toolbar = NavigationToolbar2QT(self.mol1_residual_plot, self)

        mol1_residual_gbox_glayout.addWidget(mol1_residual_toolbar)
        mol1_residual_gbox_glayout.addWidget(self.mol1_residual_plot)
        
        """"""

        """"""" mol 2 """""""

        mol2_profile_frame = QtWidgets.QFrame()
        mol2_profile_frame_glayout = QtWidgets.QGridLayout(mol2_profile_frame)

        mol2_profile_gbox = QtWidgets.QGroupBox('Molecular Profiles - 2')
        mol2_profile_gbox_glayout = QtWidgets.QGridLayout(mol2_profile_gbox)
        mol2_profile_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        mol2_profile_gbox_glayout.setSpacing(4)
        mol2_profile_gbox.setStyleSheet(groupBoxStyle)

        mol2_residual_gbox = QtWidgets.QGroupBox('Residuals')
        mol2_residual_gbox_glayout = QtWidgets.QGridLayout(mol2_residual_gbox)
        mol2_residual_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        mol2_residual_gbox_glayout.setSpacing(4)
        mol2_residual_gbox.setStyleSheet(groupBoxStyle)

        mol2_sep_profiles_label = QtWidgets.QLabel('Profile Seperation :')
        mol2_sep_redisuals_label = QtWidgets.QLabel('Residual Seperation :')

        self.mol2_sep_profiles_ledit = QtWidgets.QLineEdit(str(self.mol_sp))
        self.mol2_sep_residuals_ledit = QtWidgets.QLineEdit(str(self.mol_sr))

        self.mol2_error_bar_cbox = QtWidgets.QCheckBox('Show Errorbar')
        self.mol2_error_bar_cbox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.mol2_error_bar_cbox.setChecked(True if self.seb else False)

        mol2_replot_profiles_button = QtWidgets.QPushButton('Replot')

        mol2_profile_settings_hlayout = QtWidgets.QHBoxLayout()
        mol2_profile_settings_hlayout.addStretch()
        mol2_profile_settings_hlayout.addWidget(mol2_sep_profiles_label)
        mol2_profile_settings_hlayout.addWidget(self.mol2_sep_profiles_ledit)
        mol2_profile_settings_hlayout.addWidget(mol2_sep_redisuals_label)
        mol2_profile_settings_hlayout.addWidget(self.mol2_sep_residuals_ledit)
        mol2_profile_settings_hlayout.addWidget(self.mol2_error_bar_cbox)
        mol2_profile_settings_hlayout.addWidget(mol2_replot_profiles_button)
        mol2_profile_settings_hlayout.addStretch()

        mol2_profile_frame_glayout.addWidget(mol2_profile_gbox, 0, 0, 1, 1)
        mol2_profile_frame_glayout.addWidget(mol2_residual_gbox, 0, 1, 1, 1)
        mol2_profile_frame_glayout.addLayout(mol2_profile_settings_hlayout, 1, 0, 1, 2)

        self.mol2_profile_plot = MplCanvas(tight_layout=True)
        mol2_profile_toolbar = NavigationToolbar2QT(self.mol2_profile_plot, self)

        mol2_profile_gbox_glayout.addWidget(mol2_profile_toolbar)
        mol2_profile_gbox_glayout.addWidget(self.mol2_profile_plot)

        self.mol2_residual_plot = MplCanvas(tight_layout=True)
        mol2_residual_toolbar = NavigationToolbar2QT(self.mol2_residual_plot, self)

        mol2_residual_gbox_glayout.addWidget(mol2_residual_toolbar)
        mol2_residual_gbox_glayout.addWidget(self.mol2_residual_plot)

        """"""

        profile_tab.addTab(line_profile_frame, 'Line Profiles')
        profile_tab.addTab(mol1_profile_frame, 'Molecular Profiles - 1')
        profile_tab.addTab(mol2_profile_frame, 'Molecular Profiles - 2')

        map_frame = QtWidgets.QFrame()
        map_frame_glayout = QtWidgets.QGridLayout(map_frame)

        self.mercator_gbox = QtWidgets.QGroupBox('Mercator Projection')
        mercator_gbox_glayout = QtWidgets.QGridLayout(self.mercator_gbox)
        mercator_gbox_glayout.setContentsMargins(0, 0, 0, 0)
        mercator_gbox_glayout.setSpacing(0)
        self.mercator_gbox.setStyleSheet(groupBoxStyle)

        self.mercator_plot = MplCanvas()
        mercator_toolbar = NavigationToolbar2QT(self.mercator_plot, self)

        mercator_gbox_glayout.addWidget(mercator_toolbar)
        mercator_gbox_glayout.addWidget(self.mercator_plot)


        self.mollweide_gbox = QtWidgets.QGroupBox('Mollweide Projection')
        mollweide_gbox_glayout = QtWidgets.QGridLayout(self.mollweide_gbox)
        mollweide_gbox_glayout.setContentsMargins(0, 0, 0, 0)
        mollweide_gbox_glayout.setSpacing(0)
        self.mollweide_gbox.setStyleSheet(groupBoxStyle)

        self.mollweide_plot = MplCanvas(projection="mollweide")
        mollweide_toolbar = NavigationToolbar2QT(self.mollweide_plot, self)

        mollweide_gbox_glayout.addWidget(mollweide_toolbar)
        mollweide_gbox_glayout.addWidget(self.mollweide_plot)


        self.spherical_gbox = QtWidgets.QGroupBox('Spherical Projection')
        spherical_gbox_glayout = QtWidgets.QGridLayout(self.spherical_gbox)
        spherical_gbox_glayout.setContentsMargins(0, 0, 0, 0)
        spherical_gbox_glayout.setSpacing(0)
        self.spherical_gbox.setStyleSheet(groupBoxStyle)

        self.spherical_plot = BaseCanvas(incl=DIP.params['incl'])

        spherical_gbox_glayout.addWidget(self.spherical_plot)

        self.mercator_gbox.hide()
        self.spherical_gbox.hide()

        self.map_select_cbox = QtWidgets.QComboBox()
        self.map_select_cbox.addItems(['Mollweide Projection', 'Mercator Projection', 'Spherical Projection'])

        map_frame_glayout.addWidget(self.map_select_cbox)
        map_frame_glayout.addWidget(self.mercator_gbox)
        map_frame_glayout.addWidget(self.mollweide_gbox)
        map_frame_glayout.addWidget(self.spherical_gbox)


        map3d_fs_var_frame = QtWidgets.QFrame()
        map3d_fs_var_frame_glayout = QtWidgets.QGridLayout(map3d_fs_var_frame)

        trid_plot_gbox = QtWidgets.QGroupBox('3D Surface Map')
        trid_plot_gbox_glayout = QtWidgets.QGridLayout(trid_plot_gbox)
        trid_plot_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        trid_plot_gbox_glayout.setSpacing(4)
        trid_plot_gbox.setStyleSheet(groupBoxStyle)

        self.trid_plot_vis = Visualization()
        self.trid_plot = self.trid_plot_vis.edit_traits(parent=self, kind='subpanel').control

        trid_dis_label = QtWidgets.QLabel('Cam. Distance :')
        trid_incl_label = QtWidgets.QLabel('Inclination :')
        trid_phase_label = QtWidgets.QLabel('Phase :')

        self.trid_incl_dsbox = QtWidgets.QDoubleSpinBox()
        self.trid_incl_dsbox.setMinimum(0)
        self.trid_incl_dsbox.setMaximum(90)
        self.trid_incl_dsbox.setSingleStep(1)
        self.trid_phase_dsbox = QtWidgets.QDoubleSpinBox()
        self.trid_phase_dsbox.setMinimum(0)
        self.trid_phase_dsbox.setMaximum(1)
        self.trid_phase_dsbox.setDecimals(4)
        self.trid_phase_dsbox.setSingleStep(0.01)
        self.trid_dis_dsbox = QtWidgets.QDoubleSpinBox()
        self.trid_dis_dsbox.setValue(5)
        self.trid_dis_dsbox.setMinimum(0)
        # self.triDDisDSBox.setMaximum(1)
        self.trid_dis_dsbox.setSingleStep(1.0)

        trid_plot_gbox_glayout.addWidget(self.trid_plot, 0, 0, 1, 6)
        trid_plot_gbox_glayout.addWidget(trid_dis_label, 1, 0, 1, 1)
        trid_plot_gbox_glayout.addWidget(self.trid_dis_dsbox, 1, 1, 1, 1)
        trid_plot_gbox_glayout.addWidget(trid_incl_label, 1, 2, 1, 1)
        trid_plot_gbox_glayout.addWidget(self.trid_incl_dsbox, 1, 3, 1, 1)
        trid_plot_gbox_glayout.addWidget(trid_phase_label, 1, 4, 1, 1)
        trid_plot_gbox_glayout.addWidget(self.trid_phase_dsbox, 1, 5, 1, 1)

        fs_variation_gbox = QtWidgets.QGroupBox('fs Variation')
        fs_variation_gbox_glayout = QtWidgets.QGridLayout(fs_variation_gbox)
        fs_variation_gbox_glayout.setContentsMargins(0, 0, 0, 0)
        fs_variation_gbox_glayout.setSpacing(0)
        fs_variation_gbox.setStyleSheet(groupBoxStyle)

        self.fs_variation_plot = MplCanvas()
        fs_variation_toolbar = NavigationToolbar2QT(self.fs_variation_plot, self)

        fs_variation_gbox_glayout.addWidget(fs_variation_toolbar)
        fs_variation_gbox_glayout.addWidget(self.fs_variation_plot)

        map3d_fs_var_frame_glayout.addWidget(trid_plot_gbox, 0, 0)
        map3d_fs_var_frame_glayout.addWidget(fs_variation_gbox, 0, 1)


        results_widget = QtWidgets.QWidget()
        result_widget_glayout = QtWidgets.QGridLayout(results_widget)

        opt_result_gbox = QtWidgets.QGroupBox('Optimization Result')
        opt_result_gbox.setStyleSheet(groupBoxStyle)
        opt_result_gbox_glayout = QtWidgets.QGridLayout(opt_result_gbox)

        opt_result_table = QtWidgets.QTableWidget()
        opt_result_table.setRowCount(6)
        opt_result_table.setColumnCount(1)
        opt_result_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Value'))
        opt_result_table.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem('Chi-square'))
        opt_result_table.setVerticalHeaderItem(1, QtWidgets.QTableWidgetItem('MEM'))
        opt_result_table.setVerticalHeaderItem(2, QtWidgets.QTableWidgetItem('Final function'))
        opt_result_table.setVerticalHeaderItem(3, QtWidgets.QTableWidgetItem('Lambda'))
        opt_result_table.setVerticalHeaderItem(4, QtWidgets.QTableWidgetItem('Number of iterations'))
        opt_result_table.setVerticalHeaderItem(5, QtWidgets.QTableWidgetItem('Number of function evaluations'))
        opt_result_table.setItem(0, 0, QtWidgets.QTableWidgetItem(str(round(DIP.chisq, 6))))
        opt_result_table.setItem(1, 0, QtWidgets.QTableWidgetItem(str(round(DIP.mem, 6))))
        opt_result_table.setItem(2, 0, QtWidgets.QTableWidgetItem(str(round(DIP.ftot, 6))))
        opt_result_table.setItem(3, 0, QtWidgets.QTableWidgetItem(str(DIP.lmbd)))
        opt_result_table.setItem(4, 0, QtWidgets.QTableWidgetItem(str(self.nit)))
        opt_result_table.setItem(5, 0, QtWidgets.QTableWidgetItem(str(self.nfev)))
        opt_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        opt_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        opt_result_gbox_glayout.addWidget(opt_result_table)

        map_information_gbox = QtWidgets.QGroupBox('Surface Map Information')
        map_information_gbox.setStyleSheet(groupBoxStyle)
        map_information_gbox_glayout = QtWidgets.QGridLayout(map_information_gbox)

        map_information_table = QtWidgets.QTableWidget()
        map_information_table.setRowCount(4)
        map_information_table.setColumnCount(1)
        map_information_table.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Value'))
        map_information_table.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem('Number of surface elements'))
        map_information_table.setVerticalHeaderItem(1, QtWidgets.QTableWidgetItem('Minimum fs'))
        map_information_table.setVerticalHeaderItem(2, QtWidgets.QTableWidgetItem('Maximum fs'))
        map_information_table.setVerticalHeaderItem(3, QtWidgets.QTableWidgetItem('Total spotted area (%)'))
        map_information_table.setItem(0, 0, QtWidgets.QTableWidgetItem(str(DIP.surface_grid['noes'])))
        map_information_table.setItem(1, 0, QtWidgets.QTableWidgetItem(str(round(min(self.recons_fss), 5))))
        map_information_table.setItem(2, 0, QtWidgets.QTableWidgetItem(str(round(max(self.recons_fss), 5))))
        map_information_table.setItem(3, 0, QtWidgets.QTableWidgetItem(str(round(self.tse * 100, 6))))
        map_information_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        map_information_table.resizeColumnsToContents()

        map_information_gbox_glayout.addWidget(map_information_table)

        lamda_search_gbox = QtWidgets.QGroupBox('Lambda Search Plot')
        lamda_search_gbox.setStyleSheet(groupBoxStyle)
        lamda_search_gbox_glayout = QtWidgets.QGridLayout(lamda_search_gbox)

        self.lamda_search_plot = MplCanvas(tight_layout=True)
        lamda_search_toolbar = NavigationToolbar2QT(self.lamda_search_plot, self)

        lamda_search_gbox_glayout.addWidget(lamda_search_toolbar)
        lamda_search_gbox_glayout.addWidget(self.lamda_search_plot)


        result_widget_glayout.addWidget(opt_result_gbox, 0, 0, 1, 1)
        result_widget_glayout.addWidget(map_information_gbox, 1, 0, 1, 1)
        result_widget_glayout.addWidget(lamda_search_gbox, 0, 1, 2, 1)

        main_tab.addTab(profile_tab, 'Observations, Models and Residuals')
        main_tab.addTab(map_frame, '2D Surface Map Projections')
        main_tab.addTab(map3d_fs_var_frame, '3D Surface Map and fs Variations')
        main_tab.addTab(results_widget, 'Results')

        self.map_select_cbox.currentTextChanged.connect(self.get_2d_map)
        self.trid_phase_dsbox.valueChanged.connect(self.set_3d_plot_ori)
        self.trid_phase_dsbox.valueChanged.connect(self.plot_fs_variation)
        self.trid_incl_dsbox.valueChanged.connect(self.set_3d_plot_ori)
        self.trid_dis_dsbox.valueChanged.connect(self.set_3d_plot_ori)

        line_replot_profiles_button.clicked.connect(self.plot_line_profiles)
        mol1_replot_profiles_button.clicked.connect(self.plot_mol1_profiles)
        mol2_replot_profiles_button.clicked.connect(self.plot_mol2_profiles)
        self.line_error_bar_cbox.stateChanged.connect(self.plot_line_profiles)
        self.mol1_error_bar_cbox.stateChanged.connect(self.plot_mol1_profiles)
        self.mol2_error_bar_cbox.stateChanged.connect(self.plot_mol2_profiles)


        if self.save_maps != None:
            self.save_data_dict = {'surface_grid': self.DIP.surface_grid,
                                   'fss': self.recons_fss,
                                   'profiles': {},
                                   'line_vels': self.DIP.line_vels,
                                   'mol1_vels': self.DIP.mol1_vels,
                                   'mol2_vels': self.DIP.mol2_vels}


        if DIP.modes['line']['mode'] == 'on':
            self.plot_line_profiles()

        if DIP.modes['mol1']['mode'] == 'on':
            self.plot_mol1_profiles()

        if DIP.modes['mol2']['mode'] == 'on':
            self.plot_mol2_profiles()

        self.plot_2d_maps()
        self.plot_3d_surface()
        self.plot_fs_variation()

        if len(self.chisqs) != 0:
            self.plot_lamda_search()

        if self.save_maps != None:
            mfile = open(save_maps, 'wb')
            pickle.dump(self.save_data_dict, mfile)
            mfile.close()

    def plot_line_profiles(self):

        spl_line_slps = self.DIP.recons_result['spotless_line_slps']
        rcs_line_slps = self.DIP.recons_result['recons_line_slps']

        line_sp = float(self.line_sep_profiles_ledit.text())
        line_sr = float(self.line_sep_residuals_ledit.text())
        seb = self.line_error_bar_cbox.checkState()

        ax1 = self.line_profile_plot.fig.axes[0]
        ax2 = self.line_residual_plot.fig.axes[0]

        ax1.cla()
        ax2.cla()

        for i, phase in enumerate(spl_line_slps):

            obs_line_prf = self.DIP.line_obs_data[phase]['prf']
            spl_line_prf = spl_line_slps[phase]['prf']
            rcs_line_prf = rcs_line_slps[phase]['prf']
            obs_line_err = self.DIP.line_obs_data[phase]['errs']

            maxv = max(self.DIP.line_vels)
            maxi = max(obs_line_prf) + i * line_sp
            residual = obs_line_prf - rcs_line_prf
            maxir = np.average(residual + i * line_sr)
            xy1 = (maxv - maxv / 3.1, maxi + line_sp / 10.)
            xy2 = (maxv - maxv / 10., maxir + line_sr / 10.)

            if seb:
                ax1.errorbar(self.DIP.line_vels, obs_line_prf + i * line_sp, yerr=obs_line_err, fmt='o', color='k', ms=self.ms)
                ax2.errorbar(self.DIP.line_vels, residual + i * line_sr, yerr=obs_line_err, fmt='o', color='k', ms=self.ms)
            else:
                ax1.plot(self.DIP.line_vels, obs_line_prf + i * line_sp, 'ko', ms=self.ms)
                ax2.plot(self.DIP.line_vels, residual + i * line_sr, 'ko', ms=self.ms)

            ax1.plot(self.DIP.line_vels, spl_line_prf + i * line_sp, 'b', linewidth=self.lw, zorder=2)
            ax1.plot(self.DIP.line_vels, rcs_line_prf + i * line_sp, 'r', linewidth=self.lw, zorder=3)
            ax1.annotate(str('%0.3f' % round(phase, 3)), xy=xy1, color='g')
            ax2.annotate(str('%0.3f' % round(phase, 3)), xy=xy2, color='g')
            ax2.axhline(i * line_sr, color='r', zorder=3)

            if self.save_maps != None:
                self.save_data_dict['profiles'][phase] = {
                                                          'obs_line_prf': obs_line_prf,
                                                          'obs_line_err': obs_line_err,
                                                          'spl_line_prf': spl_line_prf,
                                                          'rcs_line_prf': rcs_line_prf,
                                                          'line_residual': residual,
                                                          }

        ax1.plot([], [], 'ko', label='Obs. Data', ms=self.ms)
        ax1.plot([], [], 'b', label='Spotless Model', linewidth=self.lw)
        ax1.plot([], [], 'r', label='Spotted Model', linewidth=self.lw)
        ax1.set_xlabel('Velocity (km/s)', fontsize=self.fs)
        ax1.set_ylabel('$\mathregular{I/I_c}$', fontsize=self.fs)
        ax1.legend()

        ax2.set_xlabel('Velocity (km/s)', fontsize=self.fs)
        ax2.set_ylabel('Residuals', fontsize=self.fs)

        ax1.tick_params(axis='both', labelsize=self.tls)
        ax2.tick_params(axis='both', labelsize=self.tls)

        self.line_profile_plot.draw()
        self.line_residual_plot.draw()

    def plot_mol1_profiles(self):

        self.plot_mol_profiles(self.mol1_sep_profiles_ledit,
                               self.mol1_sep_residuals_ledit,
                               self.mol1_error_bar_cbox,
                               self.mol1_profile_plot,
                               self.mol1_residual_plot,
                               self.DIP.mol1_obs_data,
                               self.DIP.mol1_vels,
                               np.average(self.DIP.modes['mol1']['wrange']),
                               'spotless_mol1_slps',
                               'recons_mol1_slps')

    def plot_mol2_profiles(self):

        self.plot_mol_profiles(self.mol2_sep_profiles_ledit,
                               self.mol2_sep_residuals_ledit,
                               self.mol2_error_bar_cbox,
                               self.mol2_profile_plot,
                               self.mol2_residual_plot,
                               self.DIP.mol2_obs_data,
                               self.DIP.mol2_vels,
                               np.average(self.DIP.modes['mol2']['wrange']),
                               'spotless_mol2_slps',
                               'recons_mol2_slps')

    def plot_mol_profiles(self, sep_prf_ledit, sep_res_ledit, err_cbox, prf_plot, res_plot, mol_obs_data,
                          mol_vels, mwave, wmol1, wmol2):

        spl_mol_slps = self.DIP.recons_result[wmol1]
        rcs_mol_slps = self.DIP.recons_result[wmol2]

        mol_sp = float(sep_prf_ledit.text())
        mol_sr = float(sep_res_ledit.text())
        seb = err_cbox.checkState()

        ax1 = prf_plot.fig.axes[0]
        ax2 = res_plot.fig.axes[0]

        ax1.cla()
        ax2.cla()

        mol_waves = (mol_vels * mwave) / ac.c.to(au.kilometer / au.second).value + mwave

        for i, phase in enumerate(rcs_mol_slps):

            obs_mol_prf = mol_obs_data[phase]['prf']
            obs_mol_err = mol_obs_data[phase]['errs']
            spl_mol_prf = spl_mol_slps[phase]['prf']
            rcs_mol_prf = rcs_mol_slps[phase]['prf']

            maxv = max(mol_waves)
            minv = min(mol_waves)
            maxi = np.average(obs_mol_prf[int(len(mol_waves)*0.9):]) + i * mol_sp
            residual = obs_mol_prf - rcs_mol_prf
            maxir = np.average(residual + i * mol_sr)
            # xy1 = (maxv - maxv / 3.1, maxi + mol_sp / 10.)
            xy1 = (maxv + (maxv - minv) * 0.01, maxi)
            xy2 = (maxv - (maxv - minv) * 0.01, maxir + maxir / 10.)


            if seb:
                ax1.errorbar(mol_waves, obs_mol_prf + i * mol_sp, yerr=obs_mol_err, fmt='o', color='k', ms=self.ms)
                ax2.errorbar(mol_waves, residual + i * mol_sr, yerr=obs_mol_err, fmt='o', color='k', ms=self.ms)
            else:
                ax1.plot(mol_waves, obs_mol_prf + i * mol_sp, 'k', ms=self.ms)
                ax2.plot(mol_waves, residual + i * mol_sr, 'k', ms=self.ms)

            # ax1.plot(mol_waves, spl_mol_prf + i * mol_sp, 'b', linewidth=self.lw, zorder=2)
            ax1.plot(mol_waves, rcs_mol_prf + i * mol_sp, 'r', linewidth=self.lw, zorder=3)
            ax1.annotate(str('%0.3f' % round(phase, 3)), xy=xy1, color='g')
            ax2.annotate(str('%0.3f' % round(phase, 3)), xy=xy2, color='g')
            ax2.axhline(i * mol_sr, color='r', zorder=3)

            if self.save_maps != None:
                if wmol1 == 'spotless_mol1_slps':
                    self.save_data_dict['profiles'][phase] = {
                                                              'obs_mol1_prf': obs_mol_prf,
                                                              'obs_mol1_err': obs_mol_err,
                                                              'spl_mol1_prf': spl_mol_prf,
                                                              'rcs_mol1_prf': rcs_mol_prf,
                                                              'mol1_residual': residual,
                                                              }
                if wmol1 == 'spotless_mol2_slps':
                    self.save_data_dict['profiles'][phase] = {
                                                              'obs_mol2_prf': obs_mol_prf,
                                                              'obs_mol2_err': obs_mol_err,
                                                              'spl_mol2_prf': spl_mol_prf,
                                                              'rcs_mol2_prf': rcs_mol_prf,
                                                              'mol2_residual': residual,
                                                              }

        ax1.plot([], [], 'ko', label='Obs. Data', ms=self.ms)
        ax1.plot([], [], 'r', label='Spotted Model', linewidth=self.lw)
        ax1.set_xlabel('Wavelength ($\AA$)', fontsize=self.fs)
        ax1.set_ylabel('Normalized Flux', fontsize=self.fs)
        ax1.legend()

        ax2.set_xlabel('Wavelength ($\AA$)', fontsize=self.fs)
        ax2.set_ylabel('Residuals', fontsize=self.fs)

        ax1.tick_params(axis='both', labelsize=self.tls)
        ax2.tick_params(axis='both', labelsize=self.tls)

        prf_plot.draw()
        res_plot.draw()

    def plot_2d_maps(self):

        if self.surface_grid['method'] == 'trapezoid':

            nlons = self.surface_grid['nlons'].copy()

            xlongs, xlats, crmap, cextent, mlats, mlongs = dipu.rectmap_tg2(nlons=nlons, fss=self.recons_fss)

        elif self.surface_grid['method'] == 'healpy':

            xlongs, xlats, crmap, cextent, mlats, mlongs = dipu.rectmap_hg(fss=self.recons_fss,
                                                                           nside=self.surface_grid['nside'],
                                                                           xsize=1000, ysize=500)

        self.plot_mercator(crmap=crmap, cextent=np.rad2deg(cextent))
        self.plot_mollweide(crmap=crmap, xlats=xlats, xlongs=xlongs)
        self.plot_spherical(crmap=crmap, mlats=mlats, mlongs=mlongs)

    def plot_mercator(self, crmap, cextent):

        fig = self.mercator_plot.fig
        ax = fig.axes[0]

        img = ax.imshow(crmap, cmap='gist_heat_r', aspect='equal', extent=cextent, vmin=0, vmax=1)
        # TODO: fill_between dikkat x ve y lerin değerleri otomatik hesaplanmalı
        ax.fill_between(x=[-1.782, 361.782], y1=-self.DIP.params['incl'], y2=-92.307, color='k', alpha=0.1)

        ax.set_xticks(np.arange(0, 420, 60))
        ax.set_yticks(np.arange(-90, 120, 30))
        ax.set_xlabel('Longitude ($^\circ$)', fontsize=self.fs)
        ax.set_ylabel('Latitude ($^\circ$)', fontsize=self.fs)

        ax.grid(True)

        obslong = 360 * (1.0 - self.DIP.phases)
        ax.plot([obslong, obslong], [-85, -75], 'b',  linewidth=1)
        if 0.0 in self.DIP.phases:
            ax.plot([0, 0], [-85, -75], 'b', linewidth=1)

        ax.tick_params(axis='both', labelsize=self.tls)

        cb = fig.colorbar(img, ax=ax, location='bottom', shrink=0.5)
        cb.set_label('$f_s$', fontsize=15)
        cb.ax.tick_params(labelsize=15)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        fig.tight_layout()

        if self.save_maps != None:
            self.save_data_dict['mercator_map'] = {'imshow': {
                                                                'map': crmap,
                                                                'extent': cextent,
                                                                'cmap': 'gist_heat_r',
                                                                'aspect': 'equal',
                                                                'vmin': 0.0,
                                                                'vmax': 1.0,
                                                                'xticks': np.arange(0, 420, 60),
                                                                'yticks': np.arange(-90, 120, 30)
                                                              },
                                                    'fill_between': {
                                                                      'x': [-1.782, 361.782],
                                                                      'y1': -self.DIP.params['incl'],
                                                                      'y2': -92.307,
                                                                      'alpha': 0.1
                                                                    },
                                                    'obslong': obslong
                                                }

        self.mercator_plot.draw()

    def plot_mollweide(self, crmap, xlongs, xlats):

        fig = self.mollweide_plot.fig
        ax = fig.axes[0]

        img = ax.pcolormesh(xlongs - np.pi, xlats, crmap, cmap='gist_heat_r', vmin=0, vmax=1)

        ax.fill_between(x=[-np.pi, np.pi], y1=np.deg2rad(-self.DIP.params['incl']), y2=-np.pi / 2., color='k',
                        alpha=0.1)

        ax.set_xticks(np.deg2rad(np.arange(-120, 180, 60)))
        ax.set_yticks(np.deg2rad(np.arange(-90, 120, 30)))
        xtick_labels = np.arange(60, 360, 60)
        ax.set_xticklabels(xtick_labels, zorder=15)
        ax.grid(True)

        ax.tick_params(labelsize=self.tls)
        ax.xaxis.set_label_coords(0.5, -0.100)

        obslong = 360 * (1.0 - self.DIP.phases) - 180
        ax.plot(np.deg2rad([obslong, obslong]), np.deg2rad([-30, -20]), 'b', linewidth=1)
        if 0.0 in self.DIP.phases:
            ax.plot(np.deg2rad([0.0, 0.0]), np.deg2rad([-30, -20]), 'b', linewidth=1)

        ax.set_xlabel('Longitude ($^\circ$)', fontsize=self.fs)
        ax.set_ylabel('Latitude ($^\circ$)', fontsize=self.fs)

        cb = self.mollweide_plot.fig.colorbar(img, ax=ax, location='bottom', shrink=0.5)
        cb.set_label('$f_s$', fontsize=self.fs)
        cb.ax.tick_params(labelsize=self.tls)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        fig.tight_layout()

        if self.save_maps != None:
            self.save_data_dict['mollweide_map'] = {'pcolormesh': {
                                                                'longs': xlongs - np.pi,
                                                                'lats': xlats,
                                                                'map': crmap,
                                                                'cmap': 'gist_heat_r',
                                                                'vmin': 0.0,
                                                                'vmax': 1.0,
                                                                'xticks': np.deg2rad(np.arange(-120, 180, 60)),
                                                                'yticks': np.deg2rad(np.arange(-90, 120, 30)),
                                                                'xtick_labels': np.arange(60, 360, 60)
                                                                  },
                                                    'fill_between': {
                                                                      'x': [-np.pi, np.pi],
                                                                      'y1': np.deg2rad(-self.DIP.params['incl']),
                                                                      'y2': -np.pi / 2.,
                                                                      'label_coords': (0.5, -0.100),
                                                                      'alpha': 0.1

                                                                    },
                                                    'obslong': obslong
                                                    }

        self.mollweide_plot.draw()

    def plot_spherical(self, crmap, mlats, mlongs):

        mlats = np.rad2deg(mlats)
        mlongs = np.rad2deg(mlongs)

        minp, maxp = np.min(mlats), np.max(mlats)
        minm, maxm = np.min(mlongs), np.max(mlongs)

        mr = np.linspace(minm, maxm, 11)
        pr = np.linspace(minp, maxp, 7)

        fig = self.spherical_plot.fig
        axs = self.spherical_plot.fig.axes

        ms = [self.spherical_plot.m1, self.spherical_plot.m2,
              self.spherical_plot.m3, self.spherical_plot.m4]
        for i, (m, ax) in enumerate(zip(ms, axs)):

            m.drawmeridians(mr)
            m.drawparallels(pr)

            x, y = m(mlongs, mlats)
            img = m.contourf(x, y, crmap, 100, cmap='gist_heat_r', vmin=0, vmax=1)
            ax.set_title('Phase = ' + str('%0.2f' % (i * 0.25)), pad=10)

        fig.tight_layout()

        cb = fig.colorbar(ScalarMappable(norm=img.norm, cmap=img.cmap), ax=self.spherical_plot.fig.axes,
                          ticks=range(0, 0 + 5, 5), location='right', shrink=0.75)
        cb.set_label('$f_s$', fontsize=self.fs)
        cb.ax.tick_params(labelsize=self.tls)
        cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        self.spherical_plot.draw()

    def get_2d_map(self):

        ct = self.map_select_cbox.currentText()

        if ct == 'Mercator Projection':
            self.mercator_gbox.show()
            self.mollweide_gbox.hide()
            self.spherical_gbox.hide()

        elif ct == 'Mollweide Projection':
            self.mercator_gbox.hide()
            self.mollweide_gbox.show()
            self.spherical_gbox.hide()

        elif ct == 'Spherical Projection':
            self.mercator_gbox.hide()
            self.mollweide_gbox.hide()
            self.spherical_gbox.show()

    def plot_3d_surface(self):

        fss = np.repeat(self.recons_fss, 2)

        self.tdx = []
        self.tdy = []
        self.tdz = []
        self.triangles = []
        scalars = []
        for i, xyz in enumerate(self.surface_grid['grid_xyzs']):
            scalars.append([fss[i]] * 3)
            for row in xyz:
                self.tdx.append(row[0])
                self.tdy.append(row[1])
                self.tdz.append(row[2])
            self.triangles.append((0 + i * 3, 1 + i * 3, 2 + i * 3))
        tdmap = np.hstack(scalars)

        tmesh = self.trid_plot_vis.scene.mlab.triangular_mesh(self.tdx, self.tdy, self.tdz, self.triangles,
                                                              scalars=tdmap, colormap='gist_heat',
                                                              line_width=3.0, vmin=0, vmax=1)
        tmesh.scene.y_plus_view()

        carr = tmesh.module_manager.scalar_lut_manager.lut.table.to_array()
        ncarr = carr[::-1]
        tmesh.module_manager.scalar_lut_manager.lut.table = ncarr

        cb = self.trid_plot_vis.scene.mlab.colorbar(tmesh, nb_labels=6, label_fmt='%0.1f', orientation='vertical')
        cb.label_text_property.font_family = 'times'
        cb.label_text_property.bold = 0
        cb.label_text_property.font_size = 5

        self.trid_incl_dsbox.setValue(self.DIP.params['incl'])
        self.set_3d_plot_ori()

    def set_3d_plot_ori(self):

        azimuth = -1 * (self.trid_phase_dsbox.value() * 360)
        elevation = self.trid_incl_dsbox.value()
        distance = self.trid_dis_dsbox.value()

        self.trid_plot_vis.scene.mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)

    def plot_fs_variation(self):

        ax = self.fs_variation_plot.fig.axes[0]
        ax.cla()

        phase = [self.trid_phase_dsbox.value()]
        pvf = dipu.calc_fs_variation(phases=phase, fss=self.recons_fss,
                                     areas=self.surface_grid['grid_areas'],
                                     lats=self.surface_grid['grid_lats'],
                                     longs=self.surface_grid['grid_longs'], incl=self.DIP.params['incl'])

        ax.plot(self.aphases, self.pvf_ap, 'k', ms=2)
        ax.plot(self.cphases, self.pvf_cp, 'ro', ms=4, label='Obs. Phases')
        ax.plot(phase, pvf, 'go', ms=4, label='Current Phase')
        ax.set_xlabel('Rotational Phase')
        ax.set_ylabel('$f_s$')

        ax.legend()

        if self.save_maps != None:
            self.save_data_dict['fs_variation'] = {
                                                    'aphases': self.aphases,
                                                    'pvf_ap': self.pvf_ap,
                                                    'cphases': self.cphases,
                                                    'pvf_cp': self.pvf_cp,

                                                  }

        self.fs_variation_plot.draw()

    def plot_lamda_search(self):

        ax = self.lamda_search_plot.fig.axes[0]

        ax.plot(self.chisqs, self.mems, 'ko', ms=2)
        ax.plot(self.chisqs[self.maxcurve], self.mems[self.maxcurve], 'ro', ms=4, label='Best')
        ax.set_xlabel('Chi-square')
        ax.set_ylabel('MEM')

        ax.legend()

        if self.save_maps != None:
            self.save_data_dict['lamda_search'] = {
                                                    'chisqs': self.chisqs,
                                                    'mems': self.mems,
                                                    'best_chisq': self.bestchisq,
                                                    'best_mem': self.bestmem,

                                                  }

        self.lamda_search_plot.draw()



if __name__ == '__main__':

    import sys

    app = QtWidgets.QApplication(sys.argv)

    pg = PlotGUI()

    pg.show()
    app.exec()