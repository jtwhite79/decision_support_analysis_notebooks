import os
import numpy as np
import flopy
import pyemu

def run_and_plot_hk(real):
    # replace the par values in the control file
    pst.parameter_data.loc[:,"parval1"] = pe.loc[real,pst.par_names]
    # save the updated control file
    pst.write(os.path.join(pst_helper.new_model_ws,"test.pst"))
    # run a single model run to generate the multipliers and inputs
    pyemu.os_utils.run("pestpp-ies.exe test.pst",cwd=pst_helper.new_model_ws)

    # load the arrays
    base_arr = np.log10(np.loadtxt(os.path.join(pst_helper.new_model_ws,"arr_org","hk_Layer_1.ref")))
    pp_arr = np.log10(np.loadtxt(os.path.join(pst_helper.new_model_ws,"arr_mlt","hk0.dat_pp")))
    gr_arr = np.log10(np.loadtxt(os.path.join(pst_helper.new_model_ws,"arr_mlt","hk3.dat_gr")))
    cn_arr = np.log10(np.loadtxt(os.path.join(pst_helper.new_model_ws,"arr_mlt","hk6.dat_cn")))
    in_arr = np.log10(np.loadtxt(os.path.join(pst_helper.new_model_ws,"hk_Layer_1.ref")))
    arrs = [base_arr,cn_arr,pp_arr,gr_arr,in_arr]
    
    labels = ["log10 base","log10 constant","log10 pilot points","log10 grid","log10 resulting input"]
    # mask with ibound
    ib = m.bas6.ibound[0].array
    for i,arr in enumerate(arrs):
        arr[ib==0] = np.NaN
    
    fig,axes = plt.subplots(1,5,figsize=(20,5))
    
    # work out the multiplier min and max
    vmin1 = min([np.nanmin(a) for a in arrs[1:-1]])
    vmax1 = max([np.nanmax(a) for a in arrs[1:-1]])
    
    # plot each array
    for i,(ax,arr,label) in enumerate(zip(axes,arrs,labels)):
        if i not in [0,len(arrs)-1]:  
            cb = ax.imshow(arr,vmin=vmin1,vmax=vmax1)
        else:
            cb = ax.imshow(arr)
        ax.set_title(label)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.colorbar(cb,ax=ax)
    plt.tight_layout()
    plt.show()
