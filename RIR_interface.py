import scipy.io as sio
import numpy as np

#keys [0,360) are angle in degree

class taslp_RIR_Interface():
    #(n_t60,n_angles,n_mics,rir_len)
    def __init__(self):
        self.t60_list = [round(idx,1) for idx in np.arange(0.0,1.1,0.1) if idx!=0.1]
        self.files_list = [f'HABET_SpacedOmni_8x8x3_height1.5_dist1_roomT60_{t60}.mat' for t60 in self.t60_list]
        self.scratch_dir = '/scratch/bbje/battula12/Databases/RIRs/taslp_roomdata_360_resolution_1degree/'
        self.rirs_list = self.load_all_rirs()

    
    def load_all_rirs(self):
        lst = []
        for file_name in self.files_list:
            rir = sio.loadmat(f'{self.scratch_dir}{file_name}')
            _h = rir['trainingroom']
            x = np.array(_h.tolist()).squeeze()
            x = np.reshape(x,(x.shape[0], x.shape[-1], x.shape[-2]))
            lst.append(x)
        
        return lst # list of  arrays with shape (360, 2(n_mics), rir_len)

    def get_rirs(self, t60: float, idx_list: "list integer degrees" ):
        t60_key = self.t60_list.index(t60)
        return self.rirs_list[t60_key][idx_list,:,:] #(nb_points,  2(n_mics), rir_len))


if __name__=="__main__":
    rir_interface = taslp_RIR_Interface()
    rirs = rir_interface.get_rirs(t60=0.3, idx_list=[4,8,12,15])
    print(rirs.shape)
