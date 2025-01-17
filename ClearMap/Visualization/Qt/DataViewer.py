"""
DataViewer
==========

Data viewer showing 3d data as 2d slices.

Usage
-----

.. image:: ../Static/DataViewer.jpg

Note
----
This viewer is based on the pyqtgraph package.
"""
__author__ = 'Christoph Kirst <christoph.kirst.ck@gmail.com>, Charly Rousseau <charly.rousseau@icm-institute.org>'
__license__ = 'GPLv3 - GNU General Public License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__ = 'https://idisco.info'
__download__ = 'https://www.github.com/ChristophKirst/ClearMap2'

import time
import functools as ft

import numpy as np

import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, QRect, QSize, pyqtSignal, Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QWidget, QRadioButton, QLabel, QSplitter, QApplication, QSizePolicy, QPushButton, QCheckBox

from ClearMap.Utils.utilities import runs_on_spyder
from ClearMap.IO.IO import as_source
from ClearMap.IO.Source import Source
from ClearMap.Visualization.Qt.data_viewer_luts import LUT

pg.CONFIG_OPTIONS['useOpenGL'] = False  # set to False if trouble seeing data.

if not pg.QAPP:
    pg.mkQApp()


class DataViewer(QWidget):
    mouse_clicked = pyqtSignal(int, int, int)

    DEFAULT_SCATTER_PARAMS = {
        'pen': 'red',
        'brush': 'red',
        'symbol': '+',
        'size': 10
    }

    def __init__(self, source, axis=None, scale=None, title=None, invertY=False,
                 minMax=None, screen=None, parent=None, default_lut='flame', original_orientation='zcxy', *args):

        QWidget.__init__(self, parent, *args)
        # super().__init__(self, parent, *args)

        # ## Images sources
        self.sources = []
        self.original_orientation = original_orientation
        self.n_sources = 0
        self.scroll_axis = None
        self.source_shape = None
        self.source_scale = None  # xyz scaling factors between display and real coordinates
        self.source_index = None  # The xyz center of the current view
        self.source_range_x = None
        self.source_range_y = None
        self.source_slice = None  # current slice (in scroll axis)

        self.cross = None  # cursor
        self.pals = []  # linked DataViewers
        self.scatter = None
        self.scatter_coords = None
        self.atlas = None
        self.structure_names = None

        self.z_cursor_width = 5

        self.initializeSources(source, axis=axis, scale=scale)

        # ## Gui Construction
        original_title = title
        if title is None:
            if isinstance(source, str):
                title = source
            elif isinstance(source, Source):
                title = source.location
            if title is None:
                title = 'DataViewer'
        self.setWindowTitle(title)
        self.resize(1600, 1200)

        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # image pane
        self.view = pg.ViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(invertY)

        self.graphicsView = pg.GraphicsView()
        self.graphicsView.setObjectName("GraphicsView")
        self.graphicsView.setCentralItem(self.view)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.setSizes([self.width() - 10, 10])
        self.layout.addWidget(splitter)

        image_splitter = QSplitter()
        image_splitter.setOrientation(Qt.Vertical)
        image_splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(image_splitter)

        # Image plots
        image_options = dict(clipToView=True, autoDownsample=True, autoLevels=False, useOpenGL=None)
        if self.all_colour:
            self.image_items = []
            for s in self.sources:
                slc = self.source_slice[:s.ndim]
                layer = self.color_last(s.array[slc])
                self.image_items.append(pg.ImageItem(layer, **image_options))
        else:
            self.image_items = [pg.ImageItem(s[self.source_slice[:s.ndim]], **image_options) for s in self.sources]
        for itm in self.image_items:
            itm.setRect(QRect(0, 0, self.source_range_x, self.source_range_y))
            itm.setCompositionMode(QPainter.CompositionMode_Plus)
            self.view.addItem(itm)
        self.view.setXRange(0, self.source_range_x)
        self.view.setYRange(0, self.source_range_y)

        # Slice Selector
        if original_title:
            self.slicePlot = pg.PlotWidget(title=f"""
            <html><head/><body>
            <h1 style=" margin-top:18px; margin-bottom:12px; margin-left:0px; margin-right:0px;
                       -qt-block-indent:0; text-indent:0px;">
            <span style=" font-size:xx-large; font-weight:700;">{original_title}</span></h1></body></html>
            """)
        else:
            self.slicePlot = pg.PlotWidget()

        size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)  # TODO: add option for sizepolicy
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.slicePlot.sizePolicy().hasHeightForWidth())
        self.slicePlot.setSizePolicy(size_policy)
        self.slicePlot.setMinimumSize(QSize(0, 40 + 40*bool(original_title)))
        self.slicePlot.setObjectName("roiPlot")

        self.sliceLine = pg.InfiniteLine(0, movable=True)
        self.sliceLine.setPen((255, 255, 255, 200), width=self.z_cursor_width)
        self.sliceLine.setZValue(1)
        self.slicePlot.addItem(self.sliceLine)
        self.slicePlot.hideAxis('left')

        self.slicePlot.installEventFilter(self)

        self.updateSlicer()

        self.sliceLine.sigPositionChanged.connect(self.updateSlice)

        # Axis Tools
        self.axis_buttons = []
        axis_tools_layout, axis_tools_widget = self.__setup_axes_controls()

        # coordinate label
        self.source_pointer = np.zeros(self.sources[0].ndim, dtype=int)
        self.source_label = QLabel("")
        axis_tools_layout.addWidget(self.source_label, 0, 3)

        self.graphicsView.scene().sigMouseMoved.connect(self.updateLabelFromMouseMove)

        # compose the image viewer
        image_splitter.addWidget(self.graphicsView)
        image_splitter.addWidget(self.slicePlot)
        image_splitter.addWidget(axis_tools_widget)
        image_splitter.setSizes([self.height()-35-20, 35, 20])

        # lut widgets
        self.luts = [LUT(image=i, color=c) for i, c in zip(self.image_items, self.__get_colors(default_lut))]

        lut_layout = QtWidgets.QGridLayout()

        lut_layout.setContentsMargins(0, 0, 0, 0)
        for d, lut in enumerate(self.luts):
            lut_layout.addWidget(lut, 0, d)
        lut_widget = QWidget()
        lut_widget.setLayout(lut_layout)
        lut_widget.setContentsMargins(0, 0, 0, 0)
        lut_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)  # TODO: add option for sizepolicy
        splitter.addWidget(lut_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        # update scale
        for lut in self.luts:
            lut.range_buttons[1][2].click()
        if minMax is not None:
            self.setMinMax(minMax)

        self.show()

    @property
    def space_axes(self):
        color_axis = self.color_axis
        if color_axis is None:
            color_axis = -1  # Cannot use None with == testing because of implicit cast
        return [ax for ax in range(self.sources[0].ndim) if ax != color_axis]

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel:
            angle = event.angleDelta().y()
            # steps = angle / abs(angle)
            steps = angle / 120
            self.sliceLine.setValue(self.sliceLine.value() + steps)
        return super().eventFilter(source, event)

    def __cast_source(self, source):
        if isinstance(source, tuple):
            source = list(source)
        if not isinstance(source, list):
            source = [source]
        return source

    def initializeSources(self, source, scale=None, axis=None, update=True):
        # initialize sources and axis settings
        source = self.__cast_source(source)
        self.n_sources = len(source)
        self.sources = [as_source(s) for s in source]
        for s in self.sources:
            if s.ndim == 2:
                s.shape = s.shape + (1,)  # Add empty z dimension # FIXME: see if works or need to expand_dims

        # self.__cast_bools()
        # self.__ensure_3d()

        # source shapes
        self.source_shape = self.padded_shape(self.sources[0].shape)
        for s in self.sources:
            if s.ndim > 4:
                raise RuntimeError(f'Source has {s.ndim} > 4 dimensions: {s}!')
            if s.shape != self.source_shape:
                raise RuntimeError(f'Sources shape {self.source_shape} vs {s.shape} in source {s}!')

        # slicing
        shape = list(range(self.sources[0].ndim))
        if 3 in self.sources[0].shape:  # Color image
            shape.pop(self.sources[0].shape.index(3))
        self.scroll_axis = axis if axis is not None else shape[-1]  # Default to last axis
        self.source_index = (np.array(self.source_shape, dtype=float) / 2).astype(int)

        # scaling
        scale = np.array(scale) if scale is not None else np.array([])
        self.source_scale = np.pad(scale, (0, self.sources[0].ndim - len(scale)), 'constant', constant_values=1)

        self.updateSourceRange()
        self.updateSourceSlice()

    def setSource(self, source, index='all'):  # TODO: see if could factor with __init__
        if index == 'all':
            source = self.__cast_source(source)
            if self.n_sources != len(source):
                raise RuntimeError(f'Number of sources does not match! got {len(source)}, expected {self.n_sources}')
            source = [as_source(s) for s in source]
            index = range(self.n_sources)
        else:
            s = self.sources
            s[index] = as_source(source)
            source = s
            index = [index]

        # self.__cast_bools()
        for i in index:
            s = source[i]

            if s.shape != self.source_shape:
                raise RuntimeError('Shape of sources does not match!')
            elif s.ndim < 2 or s.ndim > 4:  # FIXME: handle RGB
                raise RuntimeError(f'Sources dont have dimensions 2, 3 or 4 but {s.ndim} in source {i}!')

            if s.ndim == 4:
                layer = self.color_last(s.array[self.source_slice[:s.ndim]])
                self.image_items[i].updateImage(layer)
            else:
                if s.ndim == 2:
                    s.shape = s.shape + (1,)
                self.image_items[i].updateImage(s[self.source_slice[:s.ndim]])
        self.sources = source

    def __setup_axes_controls(self):
        axis_tools_layout = QtWidgets.QGridLayout()
        for d, ax in enumerate('xyz'):
            button = QRadioButton(ax)
            button.setMaximumWidth(50)
            axis_tools_layout.addWidget(button, 0, d)
            button.clicked.connect(ft.partial(self.setSliceAxis, d))
            self.axis_buttons.append(button)
        self.axis_buttons[self.space_axes.index(self.scroll_axis)].setChecked(True)
        axis_tools_widget = QWidget()
        axis_tools_widget.setLayout(axis_tools_layout)

        for i in range(self.n_sources):
            box = QCheckBox(f'{i}')

            box.setMaximumWidth(50)
            box.setChecked(True)
            box.stateChanged.connect(ft.partial(self.toggle_layer, i))
            axis_tools_layout.addWidget(box, 1, i)
            self.axis_buttons.append(box)

        return axis_tools_layout, axis_tools_widget

    def toggle_layer(self, i, state):
        self.image_items[i].setVisible(state == Qt.Checked)

    def __get_colors(self, default_lut):
        if self.n_sources == 1:
            cols = [default_lut]
        elif self.n_sources == 2:
            cols = ['purple', 'green']
        else:
            cols = np.array(['white', 'green', 'red', 'blue', 'purple'] * self.n_sources)[:self.n_sources]
        return cols

    def color_last(self, source):
        shape = np.array(source.shape)
        c_idx = np.where(shape == 3)[0]
        indices = np.delete(np.arange(source.ndim), c_idx[0])
        indices = np.hstack((indices, c_idx))
        return source.transpose(indices)

    def is_color(self, source):
        return source.ndim > 3 and 3 in source.shape

    @property
    def color_axis(self):
        try:
            return self.sources[0].shape.index(3)
        except ValueError:
            return None

    @property
    def all_colour(self):
        return all([self.is_color(s) for s in self.sources])

    def getXYAxes(self):  # FIXME: properties
        return [ax for ax in range(self.sources[0].ndim) if ax not in (self.scroll_axis, self.color_axis)]

    def updateSourceRange(self):
        x, y = self.getXYAxes()
        self.source_range_x = round(self.source_scale[x] * self.source_shape[x])
        self.source_range_y = round(self.source_scale[y] * self.source_shape[y])

    def updateSourceSlice(self):
        """Set the current slice of the source"""
        if self.all_colour:
            self.source_slice = [slice(None)] * 4  # TODO: check if could use self.sources[0].ndim
        else:
            self.source_slice = [slice(None)] * 3
        if self.scroll_axis:
            self.source_slice[self.scroll_axis] = self.source_index[self.scroll_axis]
        self.source_slice = tuple(self.source_slice)

    def updateSlicer(self):
        ax = self.scroll_axis
        self.slicePlot.setXRange(0, self.source_shape[ax])
        self.sliceLine.setValue(self.source_index[ax])
        stop = self.source_shape[ax] + 0.5
        self.sliceLine.setBounds([0, stop])

    def updateLabelFromMouseMove(self, event_pos):
        x, y = self.get_coords(event_pos)
        self.sync_cursors(x, y)
        self._updateCoords(x, y)

    def _updateCoords(self, x, y):
        x_axis, y_axis = self.getXYAxes()
        pos = [None] * self.sources[0].ndim
        scaled_x, scaled_y = self.scale_coords(x, x_axis, y, y_axis)
        z = self.source_index[self.scroll_axis]
        pos[x_axis] = scaled_x
        pos[y_axis] = scaled_y
        pos[self.scroll_axis] = z
        self.source_pointer = np.array(pos)
        self.updateLabel()

    def scale_coords(self, x, x_axis, y, y_axis):
        scaled_x = min(int(x / self.source_scale[x_axis]), self.source_shape[x_axis] - 1)
        scaled_y = min(int(y / self.source_scale[y_axis]), self.source_shape[y_axis] - 1)
        return scaled_x, scaled_y

    def get_coords(self, pos):
        mouse_point = self.view.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        x = min(max(0, x), self.source_range_x)
        y = min(max(0, y), self.source_range_y)
        return x, y

    def sync_cursors(self, x, y):
        if self.cross is not None:
            self.cross.set_coords([x, y])
            self.view.update()
            for pal in self.pals:
                pal.cross.set_coords([x, y])
                pal._updateCoords(x, y)
                pal.view.update()

    def updateLabel(self):
        x_axis, y_axis = self.getXYAxes()
        x, y, z = self.source_pointer[[x_axis, y_axis, self.scroll_axis]]
        xs, ys, zs = self.source_scale[[x_axis, y_axis, self.scroll_axis]]
        slc = [Ellipsis] * max(3, self.sources[0].ndim)
        slc[x_axis] = x
        slc[y_axis] = y
        slc[self.scroll_axis] = z
        slc = tuple(slc)
        if self.all_colour:
            vals = ", ".join([str(s.array[slc]) for s in self.sources])
        else:  # FIXME: check why array does not work for ndim = 3 (i.e. why we need 2 versions)
            vals = ", ".join([str(s[slc]) for s in self.sources])
        label = f"({x}, {y}, {z}) {{{x*xs:.2f}, {y*ys:.2f}, {z*zs:.2f}}} [{vals}]"
        if self.atlas is not None:
            try:
                id_ = np.asscalar(self.atlas[slc])  # Deprecated since np version 1.16
            except AttributeError:
                id_ = self.atlas[slc].item()
            label = f" <b style='color:#2d9cfc;'>{self.structure_names[id_]} ({id_})</b>" + label
        if self.parent() is None or not self.parent().objectName().lower().startswith('dataviewer'):
            label = f"<span style='font-size: 12pt; color: black'>{label}</span>"
        self.source_label.setText(label)

    def updateSlice(self):
        ax = self.scroll_axis
        index = min(max(0, int(self.sliceLine.value())), self.source_shape[ax]-1)
        if index != self.source_index[ax]:
            self.source_index[ax] = index
            self.source_slice = self.source_slice[:ax] + (index,) + self.source_slice[ax+1:]
            self.source_pointer[ax] = index
            self.updateLabel()
            self.updateImage()
            if self.scatter is not None:
                self.plot_scatter_markers(ax, index)

    def refresh(self):
        """
        Forces the plot to refresh, notably to display scatter info on top
        Returns
        -------

        """
        self.sliceLine.setValue(self.sliceLine.value() + 1)
        self.sliceLine.setValue(self.sliceLine.value() - 1)

    def setSliceAxis(self, axis):
        # old_scroll_axis = self.scroll_axis
        self.scroll_axis = self.space_axes[axis]
        self.updateSourceRange()
        self.updateSourceSlice()

        for img_itm, src in zip(self.image_items, self.sources):
            slc = self.source_slice
            if self.all_colour:
                layer = src.array[slc]
                img_itm.updateImage(self.color_last(layer))
            else:
                img_itm.updateImage(src[slc])
            img_itm.setRect(QRect(0, 0, self.source_range_x, self.source_range_y))
        self.view.setXRange(0, self.source_range_x)
        self.view.setYRange(0, self.source_range_y)

        self.updateSlicer()

    def updateImage(self):
        for img_item, src in zip(self.image_items, self.sources):
            slc = self.source_slice[:src.ndim]
            if self.all_colour:
                image = src.array[slc]
                image = self.color_last(image)
            else:
                image = src[slc]
            if image.dtype == bool:
                image = image.view('uint8')
            img_item.updateImage(image)

    def plot_scatter_markers(self, ax, index):
        self.scatter.clear()
        self.scatter_coords.axis = ax
        pos = self.scatter_coords.get_pos(index)
        if all(pos.shape):
            if self.scatter_coords.has_colours:
                self.scatter.setData(pos=pos,
                                     symbol=(self.scatter_coords.get_symbols(index)),
                                     size=10,  # FIXME: scale size as function of zoom
                                     **self.scatter_coords.get_draw_params(index))
            else:
                self.scatter.setData(pos=pos, **DataViewer.DEFAULT_SCATTER_PARAMS.copy())  # TODO: check if copy required
        try:  # FIXME: check why some markers trigger errors
            if self.scatter_coords.half_slice_thickness is not None:
                marker_params = self.scatter_coords.get_all_data(index)
                if marker_params['pos'].shape[0]:
                    self.scatter.addPoints(symbol='o', brush=pg.mkBrush((0, 0, 0, 0)),
                                           **marker_params)  # FIXME: scale size as function of zoom
        except KeyError as err:
            print(f'DataViewer error: {err}')

    def enable_mouse_clicks(self):
        self.graphicsView.scene().sigMouseClicked.connect(self.handleMouseClick)

    def handleMouseClick(self, event):
        event.accept()
        x, y = self.get_coords(event.scenePos())
        btn = event.button()
        if btn != 1:
            return

        x_axis, y_axis = self.getXYAxes()
        scaled_x, scaled_y = self.scale_coords(x, x_axis, y, y_axis)
        self.mouse_clicked.emit(scaled_x, scaled_y, self.source_index[self.scroll_axis])

    def setMinMax(self, min_max, source=0):
        self.luts[source].lut.region.setRegion(min_max)

    def padded_shape(self, shape):
        pad_size = max(3, len(shape))
        return (shape + (1,) * pad_size)[:pad_size]

    # def __cast_bools(self):
    #     for i, s in enumerate(self.sources):
    #         if s.dtype == bool:
    #             self.sources[i] = s.view('uint8')

    # def __ensure_3d(self):
    #     for i, s in enumerate(self.sources):
    #         if s.ndim == 2:
    #             s = s.view()
    #             s.shape = s.shape + (1,)
    #             self.sources[i] = s
    #         if s.ndim != 3:
    #             raise RuntimeError(f"Sources don't have dimensions 2 or 3 but {s.ndim} in source {i}!")

############################################################################################################
# ## Tests
############################################################################################################


def _test():
    img1 = np.random.rand(*(100, 80, 30))
    if not runs_on_spyder():
        pg.mkQApp()
    DataViewer(img1)
    if not runs_on_spyder():
        instance = QApplication.instance()
        instance.exec_()


if __name__ == '__main__':
    print('testing')
    _test()
    time.sleep(60)
