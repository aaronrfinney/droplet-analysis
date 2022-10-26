#!/usr/bin/env python
# coding: utf-8

# # MDAnalysis

# In[1]:


import MDAnalysis as mda

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import scipy.spatial.distance as ssd
from scipy import constants
from scipy.linalg import eig, inv
#from scipy.cluster.hierarchy import dendrogram, linkage

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout as layout




# In[2]:


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["computer modern roman"],
    "font.size": 14})
plw = 0.6
pcs = 3
pms = 3
bfillc = [0.9,0.9,0.9]
plt.rcParams['axes.linewidth'] = plw
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = plw
plt.rcParams['xtick.minor.width'] = plw
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.major.size'] = 4.5
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.width'] = plw
plt.rcParams['ytick.minor.width'] = plw
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.major.size'] = 5
plt.rcParams["figure.figsize"] = (5,4)


# In[3]:


print(mda.__version__)
#print(nv.__version__)
print(nx.__version__)


# ### Functions

# In[4]:


def minimum_image(point,cell):
    
    rc = 1.0/cell
    point = point - cell*np.rint(rc*point)
    return(point)


# In[5]:


def centre_of_mass(positions,masses):
    com = np.zeros(3)    
    for i in range(0,len(positions)):
        com += positions[i]*masses[i]
        
    return(com/np.sum(masses))


# In[6]:


def calculate_gyration_tensor(positions):
    gyration_tensor = np.zeros((3, 3))
    for i in range(0,len(positions)):
        gyration_tensor += np.outer(positions[i], positions[i])
    return gyration_tensor / len(positions)


# ### Load the LAMMPS files

# In[7]:


LAMMPSDATA = 'lmp.data'
u = mda.Universe(LAMMPSDATA, atom_style="id resid type charge x y z")
traj_list = [TRAJLIST]
u.load_new(traj_list)
print(u.trajectory)




# ### Select the atoms of interest

# In[9]:


calcium = u.select_atoms('type 1')
carbon = u.select_atoms('type 2')
ionclusters = u.select_atoms('type 1 2')
carbonate = u.select_atoms('type 2 4')
wateroxygen = u.select_atoms('type 5')
water = u.select_atoms('type 3 5')


# ### Set the atom charges and masses

# In[10]:


# Atomic masses
mca = 40.078
mc = 12.0107
mo = 15.9994
mh = 1.0079

calcium.masses = mca
carbon.masses = mc
carbonate.masses = mc + 3*mo
wateroxygen.masses = mo
water.masses = mo + 2*mh

# Gale force field charges
qca = 2.0000
qc4 = 1.123285
qo4 = -1.041095
qo2 = -0.820000
qh2 = 0.410000

calcium.charges = qca
carbon.charges = qc4
carbonate.charges = qc4 + 3*qo4
wateroxygen.charges = qo2
water.charges = qo2 + 2*qh2


# ### Loop through the frames and identify the droplet (largest cluster)

# In[48]:


from MDAnalysis.analysis.distances import contact_matrix
import MDAnalysis.transformations as trans

# Set the cutoff for first sphere coordination
trunc_distance = 4.0#irdf.bins[truncation] 

# Density profile arrays
delta = 0.1
maxd = u.dimensions[0]*2
nbins = np.int(maxd/delta)
rad_dist_calc = np.zeros(nbins)
rad_dist_carb = np.zeros(nbins)
rad_dist_wats = np.zeros(nbins)

# Scaling factors
molarity_scaling = 1e+27/constants.Avogadro
molal_scaling = 1/((len(wateroxygen)*water.masses[0]/1000))

# Plotting arrays
time = [] # simulation time
ncls = [] # number of clusters
clsz = [] # cluster sizes
dsiz = [] # droplet size
dchg = [] # droplet charge
drgy = [] # droplet radius of gyration
dasp = [] # droplet asphericity
ccat = [] # cation concentration
cani = [] # anion concentration
cdim = [] # dimer concentration
ctet = [] # tetramer concentration
chex = [] # hexamer concentration
bcat = [] # cation molality
bani = [] # anion molality

# Volume
cvol = []

# Droplet content
drop_calc = []
drop_carb = []

frames = 0

# Loop through the frames
for ts in u.trajectory[0:-1:100]:

    if ts.frame%100 == 0:
        print(ts)
    
    # Cell dimensions
    cell = ts.dimensions[0:3]
    cvol.append(cell[0]**3)    

    #============== IDENTIFY ALL OF THE CLUSTERS ===============
    
    # Generate the adjacancy matrix
    adjmatrix = contact_matrix(ionclusters.positions, cutoff=trunc_distance, returntype='numpy', box=ts.dimensions)

    # Generate a graph from the adjacency matrix
    Graph = nx.Graph(adjmatrix)
    
    # Size of connected components in the graph
    cluster_size = [len(c) for c in sorted(nx.connected_components(Graph), key=len, reverse=True)]
    largest_cluster_size = cluster_size[0]

    # Number of connected components
    number_of_clusters = len(cluster_size)    
    
    # List of nodes in every connected component
    cluster_members = [c for c in sorted(nx.connected_components(Graph), key=len, reverse=True)]
    
    #============== ANALYSE THE LARGEST CLUSTER ===============
    
    # Get the components of the largest cluster
    largest_cluster_indices = list(cluster_members[0])
    droplet = u.atoms[(ionclusters.indices[largest_cluster_indices])]
    
    # Build a new graph
    adjmatrix_droplet = contact_matrix(droplet.positions, cutoff=trunc_distance, returntype='numpy', box=ts.dimensions)
    Graph_droplet = nx.Graph(adjmatrix_droplet)     
    
    # Depth first search for connected nodes
    T = nx.dfs_tree(Graph_droplet, source=0)
    E = list(T.edges())
        
    # Reconstruct the droplet across boundaries
    pos = droplet.positions
    for i in range(0,len(E)):   
        v = pos[E[i][1]] - pos[E[i][0]]
        pos[E[i][1]] = pos[E[i][0]] + minimum_image(v,cell)            
    
    # Droplet centre of mass
    com = centre_of_mass(pos,droplet.masses)
    
    # Calculate the gyration tensor
    gyration_tensor = calculate_gyration_tensor(pos - com)

    eVals,eVecs = eig(gyration_tensor)    # Diagonalise
    eVals = eVals.real
    eVals = np.sort(eVals)

    droplet_rgyr = np.sum(eVals)**0.5                                 # Radius of gyration
    droplet_asph = 3/2*eVals[2] - droplet_rgyr**2/2                   # Asphericity
    #droplet_acyl = eVals[1] - eVals[0]                                # Acylindricity
    #droplet_kapa = 3/2 * (np.sum(eVals**2)/np.sum(eVals)**2) - 0.5    # Relative shape anisotropy
    
    # Droplet charge
    droplet_charge = droplet.total_charge()  
    
    # Droplet Ca density profiles
    natms = np.sum(droplet.types == '1' )
    drop_calc.append(natms)
    parr = pos[droplet.types == '1']
    for i in range(0,natms):
        v = parr[i] - com
        d = np.sqrt(np.dot(v,v))/delta
        rad_dist_calc[d.astype(int)] +=1
        
    # Droplet C density profiles
    natms = np.sum(droplet.types == '2' )
    drop_carb.append(natms)
    parr = pos[droplet.types == '2']
    for i in range(0,natms):
        v = parr[i] - com
        d = np.sqrt(np.dot(v,v))/delta
        rad_dist_carb[d.astype(int)] +=1
        
    # Droplet water density profiles
    #wcutoff = maxd/2
    #wsel = 'type 5 and point ' + str(com[0])+' '+str(com[1])+' '+str(com[2])+' '+wcutoff.astype(str)
    #radwats = u.select_atoms(wsel,updating=True)
    
    for i in range(0,wateroxygen.n_atoms):        
        v = wateroxygen.positions[i] - com
        v = minimum_image(v,cell)
        
        d = np.sqrt(np.dot(v,v))/delta
        rad_dist_wats[d.astype(int)] +=1
    
    
    #============== ANALYSE THE SOLUTION ===============
    
    # Get the dispersed ions
    monomer_charges = [] 
    monomer_indices = np.argwhere(np.array(cluster_size) == 1)
    
    for i in range(0,monomer_indices.size):      
        at = list(cluster_members[np.int(monomer_indices[i])]) # ion index

        monomer_charges.append(u.atoms[at].charges)
        
        
    # Count the number of positive and negative ions
    number_cations = np.sum(np.array(monomer_charges) > 0)
    number_anions = np.sum(np.array(monomer_charges) < 0)
    concn_cations = number_cations/ts.dimensions[0]**3*molarity_scaling*1000 #mM
    concn_anions = number_anions/ts.dimensions[0]**3*molarity_scaling*1000 #mM
    molal_cations = number_cations*molal_scaling*1000 #mmol
    molal_anions = number_anions*molal_scaling*1000 #mmol
    
    
    # Count the number of small associates
    dims = np.sum(np.array(cluster_size) == 2)
    tets = np.sum(np.array(cluster_size) == 4)
    hexs = np.sum(np.array(cluster_size) == 6)
    cdim.append(dims/ts.dimensions[0]**3*molarity_scaling)
    ctet.append(tets/ts.dimensions[0]**3*molarity_scaling)
    chex.append(hexs/ts.dimensions[0]**3*molarity_scaling)
    
    # Report the time series cluster information
    #print(ts.time,number_of_clusters,largest_cluster_size, droplet_charge, 
    #      droplet_rgyr, number_cations, number_anions, molal_cations,
    #     molal_anions)    

    
    # Record the data for plotting
    time.append(ts.time/1000)
    ncls.append(number_of_clusters)
    clsz.extend(cluster_size[:])
    dsiz.append(largest_cluster_size)
    dchg.append(droplet_charge)
    drgy.append(droplet_rgyr/10)
    dasp.append(droplet_asph)
    ccat.append(concn_cations)
    cani.append(concn_anions)
    bcat.append(molal_cations)
    bani.append(molal_anions)
    
    
    frames += 1


# In[12]:


binp = np.arange(0,nbins)*delta

#rad_calc /= frames
dV = 4/3*constants.pi * ((binp+delta)**3-binp**3)
rad_calc = (rad_dist_calc / frames) / dV * 1E27/constants.Avogadro
rad_carb = (rad_dist_carb / frames) / dV * 1E27/constants.Avogadro
rad_ions = (rad_calc+rad_carb)/2
rad_wats = (rad_dist_wats / frames) / dV * 1E27/constants.Avogadro


# In[13]:


binp = binp + delta/2
fig = plt.figure()
plt.xlim(0,30)
plt.ylim(0,65)
plt.plot(binp,rad_wats,'-')
plt.plot(binp,rad_ions,'-')
plt.xlabel("Distance from CoM")
plt.ylabel("Molal concentration")
plt.plot(binp,rad_wats/rad_ions,'-')
plt.savefig('density-profiles-droplet.png',dpi=100)
np.savetxt('rad_dens_calc.dat',np.column_stack([binp,rad_calc]))
np.savetxt('rad_dens_carb.dat',np.column_stack([binp,rad_carb]))
np.savetxt('rad_dens_ions.dat',np.column_stack([binp,rad_ions]))
np.savetxt('rad_dens_wats.dat',np.column_stack([binp,rad_wats]))
watperfu = np.nan_to_num(rad_wats/rad_ions, nan=0, posinf=0, neginf=0)
np.savetxt('rad_dens_watperfu.dat',np.column_stack([binp,watperfu]))


# ### Plot the cluster size distribution

# In[14]:

clsz = np.array(clsz)
histmin = np.amin(clsz)
histmax = np.amax(clsz)
xaxis = np.arange(1,histmax+1)
bins = xaxis.size

hist, bin_edges = np.histogram(clsz, density=True, bins=bins)
fig = plt.figure()
plt.ylabel("$N.p(N)$")
plt.xlabel("$N$ ions")
plt.yscale=('log')
plt.plot(xaxis,(hist/frames)*xaxis,'.',markersize=pms)
plt.savefig('csd-droplet.png',dpi=100)


# ### Ks from concentrations

# In[15]:


catcon = np.mean(ccat)/1000
cancon = np.mean(cani)/1000
dimcon = np.mean(cdim)
tetcon = np.mean(ctet)
hexcon = np.mean(chex)

print("mM cation, anion, dimer, tetramer, hexamer concentrations")
print(catcon*1000,cancon*1000,dimcon*1000,tetcon*1000,hexcon*1000)


# In[16]:


K1 = dimcon/(catcon*cancon)
print('K1',K1)
K2 = tetcon/dimcon**2
print('K2',K2)
K3 = hexcon/dimcon**3
print('K3',K3)
print("FE equivalents")
print(-2.479*np.log(K1),-2.479*np.log(K2),-2.479*np.log(K3))


# ### Plots

# In[17]:


fig, [[ax0, ax1], [ax2, ax3], [ax4, ax5]] = plt.subplots(nrows=3, ncols=2, sharex=True,figsize=(10,10))  

# Droplet size
#ax0.set_xlim([intzmin+1.5,intzmax-0.1])
#ax0.set_ylim([0,2.5])
ax0.title.set_text('Droplet size')
ax0.set_ylabel("$N$ ions")
ax0.plot(time,dsiz,'k-',linewidth=1)

# Radius of gyration
#ax1.set_xlim([intzmin,intzmax-0.1])
#ax1.set_ylim([0,1.05])
ax1.title.set_text('Droplet radius of gyration')
ax1.set_ylabel("nm")
ax1.plot(time,drgy,'k-',linewidth=1)

# Droplet net ionic charge
#ax2.set_xlim([intzmin,intzmax-0.1])
#ax2.set_ylim([-1,1])
ax2.title.set_text('Droplet net ionic charge')
ax2.set_ylabel("$e$")
ax2.plot(time,dchg,'ko',markersize=pms)

# Number of clusters
#ax3.set_xlim([intzmin,intzmax-0.1])
#ax3.set_ylim([0,2.5])
ax3.title.set_text('Number of clusters')
ax3.set_ylabel("")
ax3.plot(time,ncls,'k-',linewidth=1)

# Free ion molalities
#ax4.set_xlim([intzmin,intzmax-0.1])
#ax4.set_ylim([0,2.5])
ax4.title.set_text('Solution molalilities')
ax4.set_ylabel("mmol/kg")
ax4.plot(time[1:-1],bani[1:-1],'ko',markersize=pms, label='anions')
ax4.plot(time[1:-1],bcat[1:-1],'ro',markersize=pms, label='cations')
legend = ax4.legend(loc='upper center')
ax4.set_xlabel("time (ns)")

# Free ion molalities
#ax4.set_xlim([intzmin,intzmax-0.1])
#ax4.set_ylim([0,2.5])
ax5.title.set_text('Dispersed ions')
ax5.set_ylabel("$N$ ions")
ax5.plot(time[10:-1],np.array(bani[10:-1])/(molal_scaling*1000),'ko',markersize=pms, label='anions')
ax5.plot(time[10:-1],np.array(bcat[10:-1])/(molal_scaling*1000),'ro',markersize=pms, label='cations')
legend = ax5.legend(loc='upper center')
ax5.set_xlabel("time (ns)")

plt.savefig('timeseries-droplet.png',dpi=100)


# ### Averages

# In[37]:


print("Mean volume")
meanV = np.mean(cvol)
print(meanV,np.std(cvol))


# In[46]:


print("Mean anion molality and number w/ errors")
print(np.mean(cani),np.std(cani),np.mean(bani)/(molal_scaling*1000),np.std(bani)/(molal_scaling*1000))


# In[47]:


print("Mean cation molality and number w/ errors")
print(np.mean(ccat),np.std(ccat),np.mean(bcat)/(molal_scaling*1000),np.std(bcat)/(molal_scaling*1000))


# In[42]:


print("Mean droplet size w/ error: total; Ca; C")
print(np.mean(dsiz),np.std(dsiz))
print(np.mean(drop_calc),np.std(drop_calc))
print(np.mean(drop_carb),np.std(drop_carb))


# In[43]:


print("Mean ions not in the droplet w/ error: Ca; C")
calc_lean = calcium.n_atoms-np.mean(drop_calc)
carb_lean = carbon.n_atoms-np.mean(drop_carb)
print(calc_lean,np.std(drop_calc))
print(carb_lean,np.std(drop_carb))
scaling = 1/meanV*molarity_scaling*1000
print(calc_lean*scaling,np.std(drop_calc)*scaling)
print(carb_lean*scaling,np.std(drop_carb)*scaling)


# In[21]:


print("Mean Rgyr w/ error (nm)")
print(np.mean(drgy),np.std(drgy))
print("Mean Asphericity w/ error")
print(np.mean(dasp),np.std(dasp))


# In[22]:


print("Mean droplet charge w/ error")
print(np.mean(dchg),np.std(dchg))


# In[23]:


print("Mean Ca, C and Ow concentration in droplet core")
minb = 7.5
maxb = np.mean(drgy)*10
print("Bounds:",minb,maxb)
print(np.mean(rad_calc[np.logical_and(binp>minb, binp<maxb)]),np.std(rad_calc[np.logical_and(binp>minb, binp<maxb)]))
print(np.mean(rad_carb[np.logical_and(binp>minb, binp<maxb)]),np.std(rad_carb[np.logical_and(binp>minb, binp<maxb)]))
print(np.mean(rad_wats[np.logical_and(binp>minb, binp<maxb)]),np.std(rad_wats[np.logical_and(binp>minb, binp<maxb)]))


# In[24]:


print("Mean water per fu")
print(np.mean(watperfu[np.logical_and(binp>minb, binp<maxb)]),np.std(watperfu[np.logical_and(binp>minb, binp<maxb)]))


# In[25]:


print("Mean number of clusters")
print(np.mean(ncls),np.std(ncls))


# ### Legacy

# In[26]:

'''
com = centre_of_mass(droplet.positions,droplet.masses)

xs = droplet.positions.T[:][0]
ys = droplet.positions.T[:][1]
zs = droplet.positions.T[:][2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs,s=5)
ax.scatter(com[0],com[1],com[2],s=100)

'''
