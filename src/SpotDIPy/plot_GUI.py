import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits.basemap import Basemap
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from astropy import units as au, constants as ac
from PyQt5 import QtCore, QtWidgets


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

    def __init__(self, tight_layout=False, projection=None, kind=None, figsize=None):
        self.fig = Figure(tight_layout=tight_layout, figsize=figsize)

        if kind == '1':
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))

            self.fig.add_subplot(ax1.get_subplotspec())
            self.fig.add_subplot(ax2.get_subplotspec())

        elif kind == '2':

            gs_kw = dict(width_ratios=[1, 1], height_ratios=[0.5, 20])
            _, axd = plt.subplot_mosaic([['upper center', 'upper center'],
                                         ['left', 'right']], gridspec_kw=gs_kw, layout="constrained")

            self.fig.add_subplot(axd['left'].get_subplotspec())
            self.fig.add_subplot(axd['right'].get_subplotspec())
            self.fig.add_subplot(axd['upper center'].get_subplotspec())

        else:
            self.axes = self.fig.add_subplot(111, projection=projection)

        super(MplCanvas, self).__init__(self.fig)


class BaseCanvas(FigureCanvasQTAgg):
    def __init__(self, incl):
        self.fig = Figure()

        gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 1])
        _, axd = plt.subplot_mosaic([['upper left', 'upper right'],
                                     ['lower left', 'lower right']], gridspec_kw=gs_kw,
                                    layout="constrained")

        self.fig.add_subplot(axd['upper left'].get_subplotspec())
        self.fig.add_subplot(axd['upper right'].get_subplotspec())
        self.fig.add_subplot(axd['lower left'].get_subplotspec())
        self.fig.add_subplot(axd['lower right'].get_subplotspec())

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

    def __init__(self, DIP, plot_params):

        super().__init__()

        self.DIP = DIP

        self.ints = DIP.opt_results['ints']
        self.mapprojs = DIP.mapprojs

        self.vmin = (DIP.params['Tcool'] / DIP.params['Tphot']) ** 4
        self.vmax = (DIP.params['Thot'] / DIP.params['Tphot']) ** 4

        line_chisq = DIP.opt_stats['Chi-square for Line Profile(s)']
        mol1_chisq = DIP.opt_stats['Chi-square for Molecular(1) Profile(s)']
        mol2_chisq = DIP.opt_stats['Chi-square for Molecular(2) Profile(s)']
        lc_chisq = DIP.opt_stats['Chi-square for Light Curve Profile']
        alpha_line_chisq = DIP.opt_stats['Alpha * Line Profile(s) Chi-square']
        beta_mol1_chisq = DIP.opt_stats['Beta * Molecular(1) Profile(s) Chi-square']
        gamma_mol2_chisq = DIP.opt_stats['Gamma * Molecular(2) Profile(s) Chi-square']
        delta_lc_chisq = DIP.opt_stats['Delta * Light Curve Profile Chi-square']
        total_chisq = DIP.opt_stats['Total Weighted Chi-square']
        mem = DIP.opt_stats['Total Entropy']
        lmbd_mem = DIP.opt_stats['Lambda * Total Entropy']
        ftot = DIP.opt_stats['Loss Function Value']

        lmbd = DIP.opt_results['lmbd']
        alpha = DIP.opt_results['alpha']
        beta = DIP.opt_results['beta']
        gamma = DIP.opt_results['gamma']
        delta = DIP.opt_results['delta']
        nit = int(DIP.opt_results['nit'])
        nfev = int(DIP.opt_results['nfev'])

        csa = DIP.opt_results['total_cool_spotted_area']
        hsa = DIP.opt_results['total_hot_spotted_area']
        psa = DIP.opt_results['total_unspotted_area']

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

        self.setWindowTitle('SpotDIPy - Plot and Result Window')
        self.resize(1024, 768)

        centralWidget = QtWidgets.QWidget(self)
        centralWidgetVLayout = QtWidgets.QVBoxLayout(centralWidget)
        self.setCentralWidget(centralWidget)

        saveOMDataButton = QtWidgets.QPushButton('Save Obs. and Model Data')
        saveMaPDataButton = QtWidgets.QPushButton('Save Map Projections Data')
        saveGUIButton = QtWidgets.QPushButton('Save GUI')
        saveButtonsHLayout = QtWidgets.QHBoxLayout()
        saveButtonsHLayout.addWidget(saveOMDataButton)
        saveButtonsHLayout.addWidget(saveMaPDataButton)
        saveButtonsHLayout.addWidget(saveGUIButton)

        main_tab = QtWidgets.QTabWidget(self)

        centralWidgetVLayout.addWidget(main_tab)
        centralWidgetVLayout.addLayout(saveButtonsHLayout)

        profile_tab = QtWidgets.QTabWidget(self)

        line_profile_frame = QtWidgets.QFrame()
        line_profile_frame_glayout = QtWidgets.QGridLayout(line_profile_frame)

        line_profile_gbox = QtWidgets.QGroupBox('Line Profiles')
        line_profile_gbox_glayout = QtWidgets.QGridLayout(line_profile_gbox)
        line_profile_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        line_profile_gbox_glayout.setSpacing(4)
        line_profile_gbox.setStyleSheet(groupBoxStyle)

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
        line_profile_frame_glayout.addLayout(line_profile_settings_hlayout, 1, 0, 1, 2)

        self.line_profile_plot = MplCanvas(tight_layout=True, kind='2', figsize=None)
        line_profile_toolbar = NavigationToolbar2QT(self.line_profile_plot, self)

        line_profile_gbox_glayout.addWidget(line_profile_toolbar)
        line_profile_gbox_glayout.addWidget(self.line_profile_plot)

        """"""" lc """""""
        lc_profile_frame = QtWidgets.QFrame()
        lc_profile_frame_glayout = QtWidgets.QGridLayout(lc_profile_frame)

        lc_profile_gbox = QtWidgets.QGroupBox('Light Curve Profile')
        lc_profile_gbox_glayout = QtWidgets.QGridLayout(lc_profile_gbox)
        lc_profile_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        lc_profile_gbox_glayout.setSpacing(4)
        lc_profile_gbox.setStyleSheet(groupBoxStyle)

        lc_residual_gbox = QtWidgets.QGroupBox('Residuals')
        lc_residual_gbox_glayout = QtWidgets.QGridLayout(lc_residual_gbox)
        lc_residual_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        lc_residual_gbox_glayout.setSpacing(4)
        lc_residual_gbox.setStyleSheet(groupBoxStyle)

        self.lc_error_bar_cbox = QtWidgets.QCheckBox('Show Errorbar')
        self.lc_error_bar_cbox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.lc_error_bar_cbox.setChecked(True if self.seb else False)

        lc_profile_settings_hlayout = QtWidgets.QHBoxLayout()
        lc_profile_settings_hlayout.addStretch()
        lc_profile_settings_hlayout.addWidget(self.lc_error_bar_cbox)
        lc_profile_settings_hlayout.addStretch()

        self.lc_profile_plot = MplCanvas(tight_layout=True, kind='1')
        lc_profile_toolbar = NavigationToolbar2QT(self.lc_profile_plot, self)

        lc_profile_gbox_glayout.addWidget(lc_profile_toolbar)
        lc_profile_gbox_glayout.addWidget(self.lc_profile_plot)

        lc_profile_frame_glayout.addWidget(lc_profile_gbox, 0, 0,)
        lc_profile_frame_glayout.addLayout(lc_profile_settings_hlayout, 1, 0)

        """"""" mol 1 """""""

        mol1_profile_frame = QtWidgets.QFrame()
        mol1_profile_frame_glayout = QtWidgets.QGridLayout(mol1_profile_frame)

        mol1_profile_gbox = QtWidgets.QGroupBox('Molecular Profiles - 1')
        mol1_profile_gbox_glayout = QtWidgets.QGridLayout(mol1_profile_gbox)
        mol1_profile_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        mol1_profile_gbox_glayout.setSpacing(4)
        mol1_profile_gbox.setStyleSheet(groupBoxStyle)

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
        mol1_profile_frame_glayout.addLayout(mol1_profile_settings_hlayout, 1, 0, 1, 2)

        self.mol1_profile_plot = MplCanvas(tight_layout=True, kind='2', figsize=(2, 1))
        mol1_profile_toolbar = NavigationToolbar2QT(self.mol1_profile_plot, self)

        mol1_profile_gbox_glayout.addWidget(mol1_profile_toolbar)
        mol1_profile_gbox_glayout.addWidget(self.mol1_profile_plot)

        """"""

        """"""" mol 2 """""""

        mol2_profile_frame = QtWidgets.QFrame()
        mol2_profile_frame_glayout = QtWidgets.QGridLayout(mol2_profile_frame)

        mol2_profile_gbox = QtWidgets.QGroupBox('Molecular Profiles - 2')
        mol2_profile_gbox_glayout = QtWidgets.QGridLayout(mol2_profile_gbox)
        mol2_profile_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        mol2_profile_gbox_glayout.setSpacing(4)
        mol2_profile_gbox.setStyleSheet(groupBoxStyle)

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
        mol2_profile_frame_glayout.addLayout(mol2_profile_settings_hlayout, 1, 0, 1, 2)

        self.mol2_profile_plot = MplCanvas(tight_layout=True, kind='2', figsize=(2, 1))
        mol2_profile_toolbar = NavigationToolbar2QT(self.mol2_profile_plot, self)

        mol2_profile_gbox_glayout.addWidget(mol2_profile_toolbar)
        mol2_profile_gbox_glayout.addWidget(self.mol2_profile_plot)

        """"""

        profile_tab.addTab(line_profile_frame, 'Line Profiles')
        profile_tab.addTab(mol1_profile_frame, 'Molecular Profiles - 1')
        profile_tab.addTab(mol2_profile_frame, 'Molecular Profiles - 2')
        profile_tab.addTab(lc_profile_frame, 'Light Curve Profile')
        profile_tab.setTabVisible(0, False)
        profile_tab.setTabVisible(1, False)
        profile_tab.setTabVisible(2, False)
        profile_tab.setTabVisible(3, True)

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
        spherical_toolbar = NavigationToolbar2QT(self.spherical_plot, self)

        spherical_gbox_glayout.addWidget(spherical_toolbar)
        spherical_gbox_glayout.addWidget(self.spherical_plot)

        self.trid_plot_gbox = QtWidgets.QGroupBox('3D Surface Map')
        trid_plot_gbox_glayout = QtWidgets.QGridLayout(self.trid_plot_gbox)
        trid_plot_gbox_glayout.setContentsMargins(4, 4, 4, 4)
        trid_plot_gbox_glayout.setSpacing(4)
        self.trid_plot_gbox.setStyleSheet(groupBoxStyle)

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
        self.trid_dis_dsbox.setValue(8)
        self.trid_dis_dsbox.setMinimum(0)
        self.trid_dis_dsbox.setSingleStep(1.0)

        trid_plot_gbox_glayout.addWidget(self.trid_plot, 0, 0, 1, 6)
        trid_plot_gbox_glayout.addWidget(trid_dis_label, 1, 0, 1, 1)
        trid_plot_gbox_glayout.addWidget(self.trid_dis_dsbox, 1, 1, 1, 1)
        trid_plot_gbox_glayout.addWidget(trid_incl_label, 1, 2, 1, 1)
        trid_plot_gbox_glayout.addWidget(self.trid_incl_dsbox, 1, 3, 1, 1)
        trid_plot_gbox_glayout.addWidget(trid_phase_label, 1, 4, 1, 1)
        trid_plot_gbox_glayout.addWidget(self.trid_phase_dsbox, 1, 5, 1, 1)

        maps_tab = QtWidgets.QTabWidget(self)

        maps_tab.addTab(self.mollweide_gbox, 'Mollweide Projection')
        maps_tab.addTab(self.mercator_gbox, 'Mercator Projection')
        maps_tab.addTab(self.spherical_gbox, 'Spherical Projection')
        maps_tab.addTab(self.trid_plot_gbox, '3D Projection')

        results_tab = QtWidgets.QTabWidget(self)

        result_frame = QtWidgets.QFrame()
        result_frame_glayout = QtWidgets.QGridLayout(result_frame)

        opt_result_gbox = QtWidgets.QGroupBox('Optimization Result')
        opt_result_gbox.setStyleSheet(groupBoxStyle)
        opt_result_gbox_glayout = QtWidgets.QGridLayout(opt_result_gbox)

        opt_results_row_names = ['Chi-square for Line Profile(s)', 'Chi-square for Molecular(1) Profile(s)',
                                 'Chi-square for Molecular(2) Profile(s)', 'Chi-square for Light Curve Profile',
                                 'alpha * Line Profile(s) Chi-square', 'beta * Molecular(1) Profile(s) Chi-square',
                                 'gamma * Molecular(2) Profile(s) Chi-square', 'delta * Light Curve Profile Chi-square',
                                 'Total Chi-square', 'Total Entropy', 'lmbd * Total Entropy',
                                 'Loss Function Value', 'Number of iteration(s)', 'Number of function evaluation(s)',
                                 'alpha', 'beta', 'gamma', 'delta', 'lmbd']
        opt_results_row_vals = [line_chisq, mol1_chisq, mol2_chisq, lc_chisq, alpha_line_chisq, beta_mol1_chisq,
                                gamma_mol2_chisq, delta_lc_chisq, total_chisq, mem, lmbd_mem, ftot, nit, nfev, alpha,
                                beta, gamma, delta, lmbd]

        opt_result_table = QtWidgets.QTableWidget(len(opt_results_row_names), 1)
        opt_result_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        opt_result_table.horizontalHeader().setVisible(False)
        for rni, (row_name, row_val) in enumerate(zip(opt_results_row_names, opt_results_row_vals)):
            opt_result_table.setVerticalHeaderItem(rni, QtWidgets.QTableWidgetItem(row_name))
            row_val = row_val if type(row_val) is int else round(row_val, 6)
            opt_result_table.setItem(rni, 0, QtWidgets.QTableWidgetItem(str(row_val)))
        opt_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        opt_result_table.resizeColumnsToContents()

        opt_result_gbox_glayout.addWidget(opt_result_table)

        map_information_row_names = ['Number of Surface Elements', 'Surface Discritization Method',
                                     'Total Spotted Area (%)', 'Total Cool Spotted Area (%)',
                                     'Total Hot Spotted Area (%)', 'Total Unspotted Area (%)',
                                     'Minimum Reduced Intensity (I/Iphot)', 'Maximum Reduced Intensity (I/Iphot)']
        map_information_row_vals = [DIP.surface_grid['noes'], DIP.surface_grid['method'], round((csa + hsa) * 100, 6),
                                    round(csa * 100, 6), round(hsa * 100, 6), round(psa * 100, 6), round(self.vmin, 6),
                                    round(self.vmax, 6)]
        map_information_gbox = QtWidgets.QGroupBox('Stellar Surface Information')
        map_information_gbox.setStyleSheet(groupBoxStyle)
        map_information_gbox_glayout = QtWidgets.QGridLayout(map_information_gbox)

        map_information_table = QtWidgets.QTableWidget(len(map_information_row_names), 1)
        map_information_table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        map_information_table.horizontalHeader().setVisible(False)
        for rni, (row_name, row_val) in enumerate(zip(map_information_row_names, map_information_row_vals)):
            map_information_table.setVerticalHeaderItem(rni, QtWidgets.QTableWidgetItem(row_name))
            map_information_table.setItem(rni, 0, QtWidgets.QTableWidgetItem(str(row_val)))

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

        result_frame_glayout.addWidget(opt_result_gbox, 0, 0, 1, 1)
        result_frame_glayout.addWidget(map_information_gbox, 0, 1, 1, 1)

        results_tab.addTab(result_frame, 'Optimization Results')
        results_tab.addTab(lamda_search_gbox, 'Lambda Search')
        results_tab.setTabVisible(1, False)

        main_tab.addTab(profile_tab, 'Observations, Models and Residuals')
        main_tab.addTab(maps_tab, 'Surface Map Projections')
        main_tab.addTab(results_tab, 'Results')

        self.trid_phase_dsbox.valueChanged.connect(self.set_3d_plot_ori)
        self.trid_incl_dsbox.valueChanged.connect(self.set_3d_plot_ori)
        self.trid_dis_dsbox.valueChanged.connect(self.set_3d_plot_ori)

        line_replot_profiles_button.clicked.connect(self.plot_line_profiles)
        mol1_replot_profiles_button.clicked.connect(self.plot_mol1_profiles)
        mol2_replot_profiles_button.clicked.connect(self.plot_mol2_profiles)
        self.line_error_bar_cbox.stateChanged.connect(self.plot_line_profiles)
        self.lc_error_bar_cbox.stateChanged.connect(self.plot_lc_profile)
        self.mol1_error_bar_cbox.stateChanged.connect(self.plot_mol1_profiles)
        self.mol2_error_bar_cbox.stateChanged.connect(self.plot_mol2_profiles)

        saveOMDataButton.clicked.connect(self.save_obs_model_data)
        saveMaPDataButton.clicked.connect(self.save_map_data)
        saveGUIButton.clicked.connect(self.saveGUI)

        if 'mol2' in DIP.conf:
            profile_tab.setTabVisible(2, True)
            self.plot_mol2_profiles()
            profile_tab.setCurrentIndex(2)

        if 'mol1' in DIP.conf:
            profile_tab.setTabVisible(1, True)
            self.plot_mol1_profiles()
            profile_tab.setCurrentIndex(1)

        if 'line' in DIP.conf:
            profile_tab.setTabVisible(0, True)
            self.plot_line_profiles()
            profile_tab.setCurrentIndex(0)

        self.plot_lc_profile()
        self.plot_mercator(crmap=self.mapprojs['rmap'], cextent=np.rad2deg(self.mapprojs['extent']))
        self.plot_mollweide(crmap=self.mapprojs['rmap'], xlats=self.mapprojs['xlats'], xlongs=self.mapprojs['xlongs'])
        self.plot_spherical(crmap=self.mapprojs['rmap'], mlats=self.mapprojs['mlats'], mlongs=self.mapprojs['mlongs'])
        self.plot_3d_surface()

        if 'lmbds' in DIP.opt_results and len(DIP.opt_results['lmbds']) > 1:
            results_tab.setTabVisible(1, True)

            lmbds = DIP.opt_results['lmbds']
            chisqs = DIP.opt_results['total_wchisqs']
            mems = DIP.opt_results['mems']
            maxcurve = DIP.opt_results['maxcurve']
            self.plot_lambda_search(lmbds=lmbds, chisqs=chisqs, mems=mems, maxcurve=maxcurve)

    def save_obs_model_data(self):

        path, _ = QtWidgets.QFileDialog.getSaveFileName()

        if path:
            try:
                self.DIP.save_obs_model_data(path)
                QtWidgets.QMessageBox.information(self, 'SpotDIPy', 'The data has been saved!')
            except IOError:
                QtWidgets.QMessageBox.information(self, 'SpotDIPy', 'The data has NOT been saved!')

    def save_map_data(self):

        path, _ = QtWidgets.QFileDialog.getSaveFileName()

        if path:
            try:
                self.DIP.save_map_data(path)
                QtWidgets.QMessageBox.information(self, 'SpotDIPy', 'The data has been saved!')
            except IOError:
                QtWidgets.QMessageBox.information(self, 'SpotDIPy', 'The data has NOT been saved!')

    def saveGUI(self):

        path, _ = QtWidgets.QFileDialog.getSaveFileName()

        if path:
            try:
                file = open(path, 'wb')
                pickle.dump(self.DIP, file)
                file.close()

                QtWidgets.QMessageBox.information(self, 'SpotDIPy', 'The GUI has been saved!')
            except IOError:
                QtWidgets.QMessageBox.information(self, 'SpotDIPy', 'The GUI has NOT been saved!')

    def plot_line_profiles(self):

        spl_line_slps = self.DIP.opt_results['line']['spotless_sprfs']
        rcs_line_slps = self.DIP.opt_results['line']['recons_sprfs']

        line_sp = float(self.line_sep_profiles_ledit.text())
        line_sr = float(self.line_sep_residuals_ledit.text())
        seb = self.line_error_bar_cbox.checkState()

        fig = self.line_profile_plot.fig

        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]

        ax1.cla()
        ax2.cla()
        ax3.cla()

        ax3.axis("off")

        for i, time in enumerate(spl_line_slps):

            epoch = (time - self.DIP.params['t0']) / self.DIP.params['period']

            obs_line_prf = self.DIP.idc['line']['data'][time]['prf']
            obs_line_err = self.DIP.idc['line']['data'][time]['errs']
            spl_line_prf = spl_line_slps[time]['prf']
            rcs_line_prf = rcs_line_slps[time]['prf']

            maxv = max(self.DIP.idc['line']['vels'])
            maxi = max(obs_line_prf) + i * line_sp
            residual = obs_line_prf - rcs_line_prf
            maxir = np.average(residual + i * line_sr)
            xy1 = (maxv - maxv / 3.1, maxi + line_sp / 10.)
            xy2 = (maxv - maxv / 10., maxir + line_sr / 10.)

            if seb:
                ax1.errorbar(self.DIP.idc['line']['vels'], obs_line_prf + i * line_sp, yerr=obs_line_err, fmt='o',
                             color='k', ms=self.ms, zorder=1)
                ax2.errorbar(self.DIP.idc['line']['vels'], residual + i * line_sr, yerr=obs_line_err, fmt='o',
                             color='k', ms=self.ms, zorder=1)
            else:
                ax1.plot(self.DIP.idc['line']['vels'], obs_line_prf + i * line_sp, 'ko', ms=self.ms, zorder=1)
                ax2.plot(self.DIP.idc['line']['vels'], residual + i * line_sr, 'ko', ms=self.ms, zorder=1)

            ax1.plot(self.DIP.idc['line']['vels'], spl_line_prf + i * line_sp, 'b', linewidth=self.lw, zorder=2)
            ax1.plot(self.DIP.idc['line']['vels'], rcs_line_prf + i * line_sp, 'r', linewidth=self.lw, zorder=3)
            ax1.annotate(str('%0.3f' % round(epoch, 3)), xy=xy1, color='g')
            ax2.annotate(str('%0.3f' % round(epoch, 3)), xy=xy2, color='g')
            ax2.axhline(i * line_sr, color='r', zorder=2)

        ax1.axvline(0.0, color='g')

        ax3.plot([], [], 'ko', label='Observed Line Profiles', ms=self.ms)
        ax3.plot([], [], 'b', label='Spotless Model', linewidth=self.lw)
        ax3.plot([], [], 'r', label='Spotted Model', linewidth=self.lw)
        ax3.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False)

        ax1.set_xlabel('Velocity (km/s)', fontsize=self.fs)
        ax1.set_ylabel('$\mathregular{I/I_c}$', fontsize=self.fs)

        ax2.set_xlabel('Velocity (km/s)', fontsize=self.fs)
        ax2.set_ylabel('Residuals', fontsize=self.fs)

        ax1.tick_params(axis='both', labelsize=self.tls)
        ax2.tick_params(axis='both', labelsize=self.tls)

        fig.subplots_adjust(hspace=0)

        self.line_profile_plot.draw()

    def plot_lc_profile(self):

        seb = self.lc_error_bar_cbox.checkState()

        ax1 = self.lc_profile_plot.fig.axes[0]
        ax1.cla()

        ax2 = self.lc_profile_plot.fig.axes[1]
        ax2.cla()

        recons_slc = self.DIP.opt_results['lc']['recons_slc']

        if 'lc' in self.DIP.conf:

            times = self.DIP.idc['lc']['times']
            fluxs = self.DIP.idc['lc']['data']['fluxs']
            errs = self.DIP.idc['lc']['data']['errs']

            epochs = (times - self.DIP.params['t0']) / self.DIP.params['period']

            if seb:
                ax1.errorbar(epochs, fluxs, yerr=errs, fmt='o', color='k', label='Observed Light Curve', ms=self.ms,
                             zorder=1)
                ax2.errorbar(epochs, fluxs - recons_slc, yerr=errs, fmt='o', color='k', ms=self.ms, zorder=1)
            else:
                ax1.plot(epochs, fluxs, 'ko', label='Observed Light Curve', ms=self.ms, zorder=1)
                ax2.plot(epochs, fluxs - recons_slc, 'ko', ms=self.ms, zorder=1)

            ax1.plot(epochs, recons_slc, 'r', label='Spotted Model', linewidth=self.lw, zorder=2)

        else:
            ntimes = self.DIP.opt_results['lc']['ntimes']
            nepochs = (ntimes - self.DIP.params['t0']) / self.DIP.params['period']

            ax1.plot(nepochs, recons_slc, 'ko', ms=2, label='Synthetic Light Curve')

        ax1.set_xlabel('Epoch', fontsize=self.fs)
        ax1.set_ylabel('Normalized Flux', fontsize=self.fs)

        ax2.set_xlabel('Epoch', fontsize=self.fs)
        ax2.set_ylabel('Residuals', fontsize=self.fs)

        ax1.tick_params(axis='both', labelsize=self.tls)
        ax2.tick_params(axis='both', labelsize=self.tls)

        ax2.axhline(0.0, color='r', linewidth=self.lw, zorder=2)

        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False)

        self.lc_profile_plot.draw()

    def plot_mol1_profiles(self):

        self.plot_mol_profiles(self.mol1_sep_profiles_ledit,
                               self.mol1_sep_residuals_ledit,
                               self.mol1_error_bar_cbox,
                               self.mol1_profile_plot,
                               self.DIP.idc['mol1']['data'],
                               self.DIP.idc['mol1']['vels'],
                               np.average(self.DIP.conf['mol1']['wave_range']),
                               'mol1')

    def plot_mol2_profiles(self):

        self.plot_mol_profiles(self.mol2_sep_profiles_ledit,
                               self.mol2_sep_residuals_ledit,
                               self.mol2_error_bar_cbox,
                               self.mol2_profile_plot,
                               self.DIP.idc['mol2']['data'],
                               self.DIP.idc['mol2']['vels'],
                               np.average(self.DIP.conf['mol2']['wave_range']),
                               'mol2')

    def plot_mol_profiles(self, sep_prf_ledit, sep_res_ledit, err_cbox, prf_plot, mol_obs_data,
                          mol_vels, mwave, mode):

        rcs_mol_slps = self.DIP.opt_results[mode]['recons_sprfs']

        mol_sp = float(sep_prf_ledit.text())
        mol_sr = float(sep_res_ledit.text())
        seb = err_cbox.checkState()

        ax1 = prf_plot.fig.axes[0]
        ax2 = prf_plot.fig.axes[1]
        ax3 = prf_plot.fig.axes[2]

        ax1.cla()
        ax2.cla()
        ax3.cla()

        ax3.axis("off")

        mol_waves = (mol_vels * mwave) / ac.c.to(au.kilometer / au.second).value + mwave

        for i, time in enumerate(rcs_mol_slps):

            epoch = (time - self.DIP.params['t0']) / self.DIP.params['period']

            obs_mol_prf = mol_obs_data[time]['prf']
            obs_mol_err = mol_obs_data[time]['errs']
            rcs_mol_prf = rcs_mol_slps[time]['prf']

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

            ax1.plot(mol_waves, rcs_mol_prf + i * mol_sp, 'r', linewidth=self.lw, zorder=3)
            ax1.annotate(str('%0.3f' % round(epoch, 3)), xy=xy1, color='g')
            ax2.annotate(str('%0.3f' % round(epoch, 3)), xy=xy2, color='g')
            ax2.axhline(i * mol_sr, color='r', zorder=3)

        ax3.plot([], [], 'ko', label='Observed Molecular Profile', ms=self.ms)
        ax3.plot([], [], 'r', label='Spotted Model', linewidth=self.lw)
        ax3.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False)

        ax1.set_xlabel('Wavelength ($\AA$)', fontsize=self.fs)
        ax1.set_ylabel('Normalized Flux', fontsize=self.fs)
        # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False)

        ax2.set_xlabel('Wavelength ($\AA$)', fontsize=self.fs)
        ax2.set_ylabel('Residuals', fontsize=self.fs)

        ax1.tick_params(axis='both', labelsize=self.tls)
        ax2.tick_params(axis='both', labelsize=self.tls)

        prf_plot.draw()

    def plot_mercator(self, crmap, cextent):

        fig = self.mercator_plot.fig
        ax = fig.axes[0]

        img = ax.imshow(crmap, cmap='gist_heat', aspect='equal', extent=cextent, interpolation='bicubic',
                        vmin=self.vmin, vmax=self.vmax)

        ax.fill_between(x=[cextent[0], cextent[1]], y1=-self.DIP.params['incl'], y2=cextent[2], color='k', alpha=0.3)

        ax.set_xticks(np.arange(0, 420, 60))
        ax.set_yticks(np.arange(-90, 120, 30))
        ax.set_xlabel('Longitude ($^\circ$)', fontsize=self.fs)
        ax.set_ylabel('Latitude ($^\circ$)', fontsize=self.fs)

        ax.grid(True)

        colors = ['b', 'white', 'g', 'purple']
        for i, mode in enumerate(self.DIP.conf):
            epochs = (self.DIP.idc[mode]['times'] - self.DIP.params['t0']) / self.DIP.params['period']
            phases = epochs - np.floor(epochs)
            obslong = 360 * (1.0 - phases)

            ax.plot([obslong, obslong], [-85, -75], '-', color=colors[i], linewidth=1)
            if 0.0 in phases:
                ax.plot([0, 0], [-85, -75], '-', color=colors[i], linewidth=1)

        ax.tick_params(axis='both', labelsize=self.tls)

        cb = fig.colorbar(img, ax=ax, location='bottom', shrink=0.5)
        cb.set_label(r'$\frac{I}{I_{phot}}$', fontsize=self.fs + 5)
        cb.ax.tick_params(labelsize=self.tls)
        cb.set_ticks(np.linspace(self.vmin, self.vmax, 6))

        fig.tight_layout()

        self.mercator_plot.draw()

    def plot_mollweide(self, crmap, xlongs, xlats):

        fig = self.mollweide_plot.fig
        ax = fig.axes[0]

        img = ax.pcolormesh(xlongs - np.pi, xlats, crmap, cmap='gist_heat', shading='gouraud',
                            vmin=self.vmin, vmax=self.vmax)

        ax.fill_between(x=[-np.pi, np.pi], y1=np.deg2rad(-self.DIP.params['incl']), y2=-np.pi / 2., color='k',
                        alpha=0.3)

        ax.set_xticks(np.deg2rad(np.arange(-120, 180, 60)))
        ax.set_yticks(np.deg2rad(np.arange(-90, 120, 30)))
        xtick_labels = np.arange(60, 360, 60)
        ax.set_xticklabels(xtick_labels, zorder=15)
        ax.grid(True)

        ax.tick_params(labelsize=self.tls)
        ax.xaxis.set_label_coords(0.5, -0.100)

        colors = ['b', 'white', 'g', 'purple']
        for i, mode in enumerate(self.DIP.conf):
            epochs = (self.DIP.idc[mode]['times'] - self.DIP.params['t0']) / self.DIP.params['period']
            phases = epochs - np.floor(epochs)
            obslong = 360 * (1.0 - phases) - 180

            ax.plot(np.deg2rad([obslong, obslong]), np.deg2rad([-30, -20]), '-', color=colors[i], linewidth=1)
            if 0.0 in phases:
                ax.plot(np.deg2rad([0.0, 0.0]), np.deg2rad([-30, -20]), '-', color=colors[i], linewidth=1)

        ax.set_xlabel('Longitude ($^\circ$)', fontsize=self.fs)
        ax.set_ylabel('Latitude ($^\circ$)', fontsize=self.fs)

        cb = self.mollweide_plot.fig.colorbar(img, ax=ax, location='bottom', shrink=0.5)
        cb.set_label(r'$\frac{I}{I_{phot}}$', fontsize=self.fs + 5)
        cb.ax.tick_params(labelsize=self.tls)
        cb.set_ticks(np.linspace(self.vmin, self.vmax, 6))

        fig.tight_layout()

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

            m.drawmeridians(mr, linewidth=1.0)
            m.drawparallels(pr, linewidth=1.0)

            x, y = m(mlongs, mlats)
            img = m.contourf(x, y, crmap, 100, cmap='gist_heat', vmin=self.vmin, vmax=self.vmax)
            ax.set_title('Phase = ' + str('%0.2f' % (i * 0.25)), pad=10)

        fig.tight_layout()

        cb = fig.colorbar(ScalarMappable(norm=img.norm, cmap=img.cmap), ax=self.spherical_plot.fig.axes,
                          ticks=range(0, 0 + 5, 5), location='right', shrink=0.75)
        cb.set_label(r'$\frac{I}{I_{phot}}$', fontsize=self.fs + 5)
        cb.ax.tick_params(labelsize=self.tls)
        cb.set_ticks(np.linspace(self.vmin, self.vmax, 6))

        self.spherical_plot.draw()

    def plot_3d_surface(self):

        fss = self.ints.copy()
        grid_xyzs = self.surface_grid['grid_xyzs'].copy()
        if self.DIP.surface_grid['method'] != 'phoebe2_marching':
            fss = np.repeat(self.ints, 2)

        tdx = []
        tdy = []
        tdz = []
        triangles = []
        scalars = []
        for i, xyz in enumerate(grid_xyzs):
            scalars.append([fss[i]] * 3)
            for row in xyz:
                tdx.append(row[0])
                tdy.append(row[1])
                tdz.append(row[2])
            triangles.append((0 + i * 3, 1 + i * 3, 2 + i * 3))
        tdmap = np.hstack(scalars)

        tmesh = self.trid_plot_vis.scene.mlab.triangular_mesh(tdx, tdy, tdz, triangles,  scalars=tdmap,
                                                              colormap='gist_heat', line_width=3.0, vmin=self.vmin,
                                                              vmax=self.vmax)

        tmesh.scene.y_plus_view()

        cb = self.trid_plot_vis.scene.mlab.colorbar(tmesh, nb_labels=6, label_fmt='%0.1f', orientation='vertical')
        cb.label_text_property.font_family = 'times'
        cb.label_text_property.bold = 0
        cb.label_text_property.font_size = 2

        self.trid_incl_dsbox.setValue(self.DIP.params['incl'])
        self.set_3d_plot_ori()

    def set_3d_plot_ori(self):

        azimuth = -1 * (self.trid_phase_dsbox.value() * 360)
        elevation = self.trid_incl_dsbox.value()
        distance = self.trid_dis_dsbox.value()

        self.trid_plot_vis.scene.mlab.view(azimuth=azimuth, elevation=elevation, distance=distance)

    def plot_lambda_search(self, lmbds, chisqs, mems, maxcurve):

        ax1 = self.lamda_search_plot.fig.axes[0]

        best_lmbd = lmbds[maxcurve]

        ax1.plot(chisqs, mems, 'ko', ms=self.ms)
        ax1.plot(chisqs[maxcurve], mems[maxcurve], 'ro', ms=self.ms + 1,
                 label='Best lmbd=' + str(best_lmbd))
        ax1.set_xlabel('$\chi^2$', fontsize=self.fs)
        ax1.set_ylabel('Entropy', fontsize=self.fs)

        ax1.tick_params(axis='both', labelsize=self.tls)
        ax1.tick_params(axis='both', labelsize=self.tls)

        ax1.legend()

        self.lamda_search_plot.draw()
