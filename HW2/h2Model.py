# Imports 
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

# Class with the Truss Model
class trussModel():

  # Construct Model
  def __init__(self):

    # Totals
    self.totNodes = 10
    self.totTruss = 17
    self.nodeDOFs = 2

    # Node Coords in SI
    self.nodes = np.array([[0.0,0.0],\
	                        [1.0,0.0],\
	                        [2.0,0.0],\
	                        [3.0,0.0],\
	                        [4.0,0.0],\
	                        [4.0,1.0],\
	                        [3.0,1.0],\
	                        [2.0,1.0],\
	                        [1.0,1.0],\
	                        [0.0,1.0]],dtype=np.float)

    # Connectivities: Lower Chord, Vertical Trusses, Diagonals, Upper Chord
    self.conn = np.array([[0,1],[1,2],[2,3],[3,4],\
	                        [0,9],[1,8],[2,7],[3,6],[4,5],\
	                        [9,1],[8,2],[6,2],[5,3],\
	                        [9,8],[8,7],[7,6],[6,5]],dtype=int)

  # Determine the length of the truss elements
  def formTrussLengths(self):
    res = np.zeros(self.totTruss)
    for loopA in range(self.totTruss):  
      node1 = self.conn[loopA,0]
      node2 = self.conn[loopA,1]
      res[loopA] = np.sqrt((self.nodes[node2,0] - self.nodes[node1,0])**2 + (self.nodes[node2,1] - self.nodes[node1,1])**2)
    return res

  # Get Rotation Matrix
  def getRotationMatrix(self,truss):
    res = np.zeros((2,4))
    node1 = self.conn[truss,0]
    node2 = self.conn[truss,1]
    # Get the versors for the coordinate transformation
    axis1 = np.array([[self.nodes[node2,0] - self.nodes[node1,0]],
                      [self.nodes[node2,1] - self.nodes[node1,1]]],dtype=np.float)
    axis1 = axis1/np.linalg.norm(axis1,2)
    # Get the versors for the coordinate transformation
    res[0,0]   =  axis1[0]
    res[0,1]   =  axis1[1]
    res[1,2]   =  axis1[0]
    res[1,3]   =  axis1[1]
    # Return
    return res

  # Get Degrees Of Freedom Mask
  def getDofMask(self,loopTruss):
    return np.array([self.conn[loopTruss,0]*2 + 0,self.conn[loopTruss,0]*2 + 1,self.conn[loopTruss,1]*2 + 0,self.conn[loopTruss,1]*2 + 1],dtype=int)

  # Function to Solve the model and evaluate the mid span vertical displacement
  def solve(self,young,area):

    # Determine Element Lengths
    elLengths = self.formTrussLengths()
    
    # Create Local Stiffness
    locStif = np.zeros((self.totTruss,2,2))
    locStif[:,0,0] =  1.0 * young * area / elLengths
    locStif[:,1,0] = -1.0 * young * area / elLengths
    locStif[:,0,1] = -1.0 * young * area / elLengths
    locStif[:,1,1] =  1.0 * young * area / elLengths

    # Assemble Contributions
    globStif = np.zeros((self.totNodes*self.nodeDOFs,self.totNodes*self.nodeDOFs))

    # Element Loop
    for loopTruss in range(self.totTruss):
      # Compute the Rotation Matrix
      rotMat = self.getRotationMatrix(loopTruss)

      # Determine Dof Mask for element
      elDofMask = self.getDofMask(loopTruss)

      # Rotate Local Stiffness 
      rotLocStif = np.dot(np.dot(rotMat.T,locStif[loopTruss,:,:]),rotMat)

      # Assemble into global stiffness
      for loopA in range(len(elDofMask)):
        rowIdx = elDofMask[loopA]
        for loopB in range(len(elDofMask)):
          colIdx = elDofMask[loopB]
          globStif[rowIdx,colIdx] += rotLocStif[loopA,loopB]

    #plt.spy(globStif, precision=0.1, markersize=10)
    #plt.show()

    # Apply Statically determinate restraint conditions
    fixedDofs = (1,8,9)
    # Remove Rows and Columns 1,8,9    
    globStif = np.delete(globStif, fixedDofs, axis=0)
    globStif = np.delete(globStif, fixedDofs, axis=1)

    # Apply load (constant load on the upper chord)
    f = np.zeros(self.totNodes*self.nodeDOFs)
    f[2*5+1] = -1.0e6
    f[2*6+1] = -1.0e6
    f[2*7+1] = -1.0e6
    f[2*8+1] = -1.0e6
    f[2*9+1] = -1.0e6
    f = np.delete(f, fixedDofs)
    activeDofs = np.arange(2*self.totNodes)
    activeDofs = np.asarray(list(set(activeDofs) - set(fixedDofs)),dtype=int)
    res = np.zeros(self.totNodes*self.nodeDOFs)

    # Solve For Displacements
    res[activeDofs] = np.linalg.solve(globStif,f)
    return np.array([res[::2],res[1::2]])

  # Plot Model To Screen
  def plotModel(self,disps):
    disps = np.reshape(disps.T,(self.totNodes,2))
    plt.figure(figsize=(8,2))
    ax = plt.subplot(111)
    for loopA in range(self.totTruss):
      x1 = self.nodes[self.conn[loopA,0],0]
      x2 = self.nodes[self.conn[loopA,1],0]
      y1 = self.nodes[self.conn[loopA,0],1]
      y2 = self.nodes[self.conn[loopA,1],1]
      currLine = np.array([(x1,y1),(x2,y2)])
      ax.plot((x1,x2),(y1,y2),'k',lw=2,alpha=0.6)
      #ax.set_xlim([-0.5,5.0])
      #ax.set_ylim([-0.5,1.5])
    for loopA in range(self.totTruss):
      x1 = self.nodes[self.conn[loopA,0],0] + disps[self.conn[loopA,0],0]
      x2 = self.nodes[self.conn[loopA,1],0] + disps[self.conn[loopA,1],0]
      y1 = self.nodes[self.conn[loopA,0],1] + disps[self.conn[loopA,0],1]
      y2 = self.nodes[self.conn[loopA,1],1] + disps[self.conn[loopA,1],1]
      currLine = np.array([(x1,y1),(x2,y2)])
      ax.plot((x1,x2),(y1,y2),'r',lw=2,alpha=0.6)
    #plt.axis('equal')
    plt.xlim([-0.5,4.5])
    plt.ylim([-0.5,1.5])
    plt.savefig('truss.svg')
    plt.show()

# =============
# MAIN FUNCTION
# =============
if __name__ == "__main__":

  # Create Truss Model
  tm = trussModel()

  # Create Random Properties
  young = np.random.uniform(190.0e9,210.0e9,(tm.totTruss))
  area = np.random.uniform(12.0e-4,16.0e-4,(tm.totTruss))
   
  # Solve Model
  disps = tm.solve(young,area)

  # Plot Result
  tm.plotModel(disps)

  # Get Centernode Disp
  print(disps[1,2])

