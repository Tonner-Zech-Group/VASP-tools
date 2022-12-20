import ase, ase.io
import pymatgen
import numpy as np
from ase import Atoms
import ase
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pymatgen.io.vasp
import ase.io
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import ndimage
from pathos.multiprocessing import ProcessingPool as Pool
import math
from ase.neighborlist import NeighborList
from ase.data import covalent_radii,colors
import json
mpl.use('TkAgg')
class STM():
    def __init__(self,filename):
        self.filename=filename
        self.data=[]
        self.atoms=ase.Atoms(None)
        self.on_grid=False
        self.nx=0;self.ny=0;self.nz=0;self.dz=0;self.dx=0;self.dy=0; #information about cube and cell
        self.height=20 #constant height
        self.rho= 0.0001 #constant current
        self.zcut=0; #constant current # ignore this
        self.missed=0; #constant current. where are we in the data
        self.new_nx=0;self.new_ny=0;self.XARRAY=[];self.YARRAY=[] #real space projection
        self.repeat=[1,1]
        self.shift=[0,0]
        self.real_space_shift=[0,0]
        self.imdata=[]
        self.colormap='Greys'
        self.show_atoms=True
        self.color_dict=None
        self.plot_atom_cutoff = 0
        self.plot_atom_colorscaling = True
        self.show_unit_cell=True
        self.show_bonds=True
        self.show_colorbar=True
        self.bond_radius=1
        self.unit_cell_ls='dashed'
        self.scale=2500 # I determined this to be a nice size. adjust if necessary.
        self.name='STM'
        self.show=True
        self.use_bondorders=True
        self.bondatoms=[]
        self.verbosity=True
        self.zoom=2
        self.aa=False
        self.mode='constant_current'
    def real_space_projection(self):
        self.new_nx=len(self.imdata[:,1])
        self.new_ny=len(self.imdata[1,:])
        self.XARRAY=np.empty(self.imdata.shape)
        self.YARRAY=np.empty(self.imdata.shape)
        for i in range(self.new_nx):
            for j in range(self.new_ny):
                x=(i)/(self.new_nx-1)
                y=(j)/(self.new_ny-1)
                v1=np.linalg.norm(self.cell[0])
                v2=np.linalg.norm(self.cell[1])
                a=self.cell[0,:-2]*self.repeat[1]
                b=self.cell[1,:-2]*self.repeat[0]
                #mat=np.array([[cellv_x,1],[cellv_y*math.cos(angle),cellv_y*math.sin(angle)]])
                #mat  = np.array([[a,0],[b*np.cos(angle),b*np.sin(angle)]])
                #A=np.vstack([a, b]).T
                #A_inv = np.linalg.inv(A)
                self.cell_2D=[self.cell[0,:-1]*self.repeat[1],self.cell[1,:-1]*self.repeat[0]]
                mult=np.matmul(np.array([x,y]),self.cell_2D)
                x_val=float(mult[0])
                y_val=float(mult[1])
                self.XARRAY[i,j]=x_val
                self.YARRAY[i,j]=y_val
    def constant_current(self,**kwargs):
        #self.data = self.data[:,:,::-1]
        self.zmax = self.nz*self.dz - self.zcut
        plane = np.empty(self.data.shape[0:2])
        self.zmax_grid=int(round(self.zmax/self.dz,0))
        print(self.zmax,self.zmax_grid)
        for i in range(self.nx):
            for j in range(self.ny):
                # argmax returns index of first occurence of maximum value
                itmp = np.argmax(self.data[i,j,:] > self.rho)
                #if itmp == 0 or itmp * self.dz > self.zmax:
                if itmp * self.dz < 1 or itmp * self.dz > self.zmax:
                   plane[i,j] = self.zmax
                   self.missed = self.missed + 1
                elif self.on_grid:
                    plane[i,j] = itmp * self.dz
                else:
                    greater = self.data[i,j,itmp]
                    smaller = self.data[i,j,itmp-1]
                    plane[i,j] = self.dz * (itmp - (greater - self.rho)/(greater-smaller))
        print(len(self.data[i,j,:self.zmax_grid]))
        plane = self.dz*self.nz - plane
        print(plane)
        #self.data=self.data[:,:,::-1]
        if self.verbosity == True:print("{} z-values replaced by zcut = {}".format(self.missed,self.zcut))
        self.imdata=plane
        repeat=[self.repeat[1],self.repeat[0]]
        self.imdata = np.tile(self.imdata, repeat)
    def constant_height(self,**kwargs):
        plane = np.empty(self.data.shape[0:2])
        self.height=int(round(self.height/self.dz,0))
        if self.verbosity == True:print(f'Height in grid-data: {self.height}')
        for i in range(self.nx):
            for j in range(self.ny):
                plane[i,j] = self.data[i,j,self.height]
    #            X=(i+1)*atoms.cell[0,0]/self.nx+(j+1)*self.cell[0,1]/self.ny
     #           Y=(i+1)*atoms.cell[1,0]/self.nx+(j+1)*self.cell[1,1]/self.ny
        self.imdata=plane
        repeat=[self.repeat[1],self.repeat[0]]
        self.imdata = np.tile(self.imdata, repeat)
    def STM_plot(self):
        def bonds(self):
            cutoffs = self.bond_radius * covalent_radii[self.new_atoms.numbers]
            nl = NeighborList(cutoffs=cutoffs, self_interaction=False,bothways=True)
            nl.update(self.new_atoms)
            self.bondatoms = []
            for a in range(len(self.new_atoms)):
                indices, offsets = nl.get_neighbors(a)
                for a2, offset in zip(indices, offsets):
                    if self.new_atoms[a].symbol == self.new_atoms[a2].symbol and self.new_atoms.get_distance(a, a2) <= 1.4:
                        bondorder=2
                        bondoffset=(0.1, 0.1, 0)
                        if self.new_atoms[a].symbol == self.new_atoms[a2].symbol and self.new_atoms.get_distance(a, a2) <= 1.34:
                            bondorder=3
                            bondoffset=(0.07, 0.07, 0)
                    else:
                        bondorder=1
                        bondoffset=(0, 0, 0)
                    if (a,a2) not in self.bondatoms:
                        if self.use_bondorders == True:
                            self.bondatoms.append((a, a2,offset,bondorder,bondoffset))
                        else:
                            self.bondatoms.append((a, a2,offset))
        def plot_atoms(self):
            self.cell_2D=[self.cell[0,:-1],self.cell[1,:-1]]
            if self.verbosity == True:
                print(f'plot atoms {self.show_atoms};plot bonds {self.show_bonds};plot the unit cell {self.show_unit_cell}')
            self.new_atoms=Atoms(None,cell=self.atoms.cell)
            [self.new_atoms.append(a) for a in self.atoms if a.position[2] >= float(self.plot_atom_cutoff)]
            self.new_atoms.pbc=self.atoms.pbc
            if self.show_unit_cell == True:
                X=[0,1,1,0,0]
                Y=[0,0,1,1,0]
                Xs=[]
                Ys=[]
                for x,y in zip(X,Y):
                    mult=np.matmul(np.array([x+self.shift[0],y+self.shift[1]]),np.array(self.cell_2D))
                    Xs.append(mult[0])
                    Ys.append(mult[1])
                self.ax.plot(Xs,Ys,linestyle='dashed',lw=1.5,color='black',zorder=10)
            if self.show_bonds == True:
                bonds(self)
                self.real_space_shift=np.matmul(np.array([self.shift[0],self.shift[1]]),np.array(self.cell_2D))
                for a1,a2,offset,bondorder,bondorderoffset in self.bondatoms:
                    atompos1=self.new_atoms[a1].position[:-1]+self.real_space_shift
                    OFFSET_REAL=np.matmul(offset[:-1],np.array(self.cell_2D))
                    atompos2=self.new_atoms[a2].position[:-1]+self.real_space_shift
                    mida = 0.5 * (atompos1 + atompos2 + OFFSET_REAL)
                    midb = 0.5 * (atompos1 + atompos2 - OFFSET_REAL)
                    xatom1=[mida[0],atompos1[0]]
                    yatom1=[mida[1],atompos1[1]]
                    xatom2=[atompos2[0],midb[0]]
                    yatom2=[atompos2[1],midb[1]]
                    self.ax.plot(xatom1,yatom1,color='black')
                    self.ax.plot(xatom2,yatom2,color='black')
            if self.show_atoms == True:
                Xs=[x+self.real_space_shift[0] for x,y,z in self.new_atoms.positions]
                Ys=[y+self.real_space_shift[1] for x,y,z in self.new_atoms.positions]
                self.COLORS=[]
                self.sizes=[]
                for n,a in enumerate(self.new_atoms.get_atomic_numbers()):
                    if self.color_dict == None:
                        colorval=colors.jmol_colors[a]
                        if self.plot_atom_colorscaling == True:
                            colorval=colorval*self.new_atoms[n].position[-1]/(np.linalg.norm(np.amax(self.new_atoms.positions[:,2])))
                        self.COLORS.append(colorval)
                    else:
                        self.COLORS=[self.color_dict[s] for s in self.new_atoms.get_chemical_symbols()]
                    sizeval=covalent_radii[a]*self.scale/np.amax(self.XARRAY) 
                    self.sizes.append(sizeval)
                self.ax.scatter(Xs,Ys,color=self.COLORS,s=self.sizes,linewidth=0.5,edgecolors='black',zorder=10) #plot the atoms. 
        self.imdata=ndimage.zoom(self.imdata, self.zoom, order=3) 
        self.XARRAY=ndimage.zoom(self.XARRAY, self.zoom, order=3)
        self.YARRAY=ndimage.zoom(self.YARRAY, self.zoom, order=3)
        self.fig=plt.figure()
        self.ax=plt.gca()
        self.ax.set_aspect('equal')
        self.fig.patch.set_facecolor('white')
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
        plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=False)
        plt.axis('off')
        self.cax=self.ax.pcolor(self.XARRAY,self.YARRAY,self.imdata,cmap=self.colormap,aa=self.aa)
        if self.show_atoms == True:
            plot_atoms(self) 
        if self.show_colorbar == True:
                divider = make_axes_locatable(self.ax)
                cax2 = divider.append_axes("right", size="5%", pad=0.15)
                if self.mode == 'constant_height':
                    cbar = plt.colorbar(self.cax,cax=cax2, format='%.1E',)
                    cbar.set_label(r'$\rho\,$[a.u.]')
                if self.mode == 'constant_current':
                    cbar = plt.colorbar(self.cax,cax=cax2, format='%.1f',)
                    cbar.set_label(r'$z\,$[\AA]')
        print(self.show)
        if self.show == True:
            plt.show()
        else:
            plt.savefig(f'{self.name}.png',transparent=False,dpi=500)
            plt.close()
    def read_data(self):
        chg=pymatgen.io.vasp.outputs.Chgcar.from_file(self.filename)
        self.atoms=AseAtomsAdaptor.get_atoms(chg.structure)
        self.cell=self.atoms.cell
        self.cell_values=[np.linalg.norm(vec) for vec in self.cell]
        self.data=np.array(chg.data['total']/np.prod(chg.data['total'].shape))
        shape=self.data.shape
        self.nx=shape[0];self.ny=shape[1];self.nz=shape[2]
        self.dx=self.cell_values[0]/self.nx;  self.dy=self.cell_values[1]/self.ny;  self.dz=self.cell_values[2]/self.nz
        self.data = self.data[:,:,::-1]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot simulated STM image from Tersoff-Hamann')
    parser.add_argument('PARCHG',type=str,help='PARCHG file from STM simulation with VASP')
    parser.add_argument('-v',help='Verbosity',action='store_true')
    parser.add_argument('-m','--mode',default='constant_current',choices=['constant_current','constant_height'])
    parser.add_argument('-H','--height',default=20,help='height for the constant height mode')
    parser.add_argument('--rho',default=0.001,help='Cutoff density for constant current (the height at this value will be plotted)')
    parser.add_argument('-c','--colormap',default='Greys_r',help='matplotlib colormap names; e.g. Afmhot,Grays')
    parser.add_argument('-r','--repeat',default='11',help='repetition of the unit cell in [x,y] direction')
    parser.add_argument('-s','--shift',default='00',help='shift the atom/unitcell plot in the units of cells. Only use with repeat >= 3')
    parser.add_argument('-a','--atoms', help='plot the atoms',action='store_true')
    parser.add_argument('-b','--bonds',help='plot the atoms but not the bonds',action='store_true')
    parser.add_argument('-u','--unitcell',help='decide if you want to plot the atoms but not the unit cell',action='store_true')
    parser.add_argument('-C','--colorbar',help='decide if you want to plot the colorbar',action='store_true')
    parser.add_argument('-S','--show',help='decide if you only want to save or show the figure',action='store_true')
    parser.add_argument('-N','--name',default='STM',help='name for the saved plot if show == False')
    parser.add_argument('--scale',default=2500,help='empiric scaling value for the atoms in the plot')
    parser.add_argument('--unit_cell_linestyle',default='dashed')
    parser.add_argument('--cutoff_radius',default=0.8,help='cutoff radius for drawing the bonds')
    parser.add_argument('--atom_colorscaling',help='scale the color of the atoms with height (lower == darker)',action='store_true')
    parser.add_argument('--atom_height_cutoff',default=0,help='Only atoms above this value will be plotted')
    parser.add_argument('--color_dict',default=None,help='This is not implemented. Use Custom colordict for atoms. Will implement soon')
    parser.add_argument('-z','--zoom',default=2,help='Interpolation of data points. Careful for large data set.')
    a = parser.parse_args()
    stm=STM(a.PARCHG)
    stm.rho=float(a.rho);stm.height=float(a.height) #science
    stm.colormap=str(a.colormap);stm.repeat=[int(a.repeat[0]),int(a.repeat[1])];stm.shift=[int(a.shift[0]),int(a.shift[1])]; #the STM
    stm.show_atoms=(a.atoms);stm.show_bonds=a.bonds;stm.show_unit_cell=a.unitcell;stm.show_colorbar=a.colorbar; #show things
    stm.plot_atom_colorscaling=a.atom_colorscaling;stm.plot_atom_cutoff=float(a.atom_height_cutoff);stm.bond_radius=float(a.cutoff_radius);stm.scale=float(a.scale);stm.unit_cell_ls=a.unit_cell_linestyle;stm.color_dict=json.loads(a.color_dict);stm.zoom=int(a.zoom) #modification
    stm.show=a.show ;stm.name=a.name #show or save
    print(a.show,stm.show)
    stm.mode=a.mode
    stm.read_data()
    if a.v==True: print('data read')
    if a.mode == 'constant_current':
        stm.constant_current()
    elif a.mode == 'constant_height':
        stm.constant_height()
    else:
        print('this mode does not exist')
        exit(-1)
    stm.real_space_projection()
    if a.v==True:print('Projection is done')
    stm.STM_plot()
    if a.v==True:print('STM_plot is done')
if __name__ == "__main__":
    main()
