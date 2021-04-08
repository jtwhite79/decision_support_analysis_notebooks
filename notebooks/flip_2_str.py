import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
import pyemu

org_d = os.path.join("temp_history")
new_d = org_d+"_str"
m = flopy.modflow.Modflow.load("freyberg.nam",model_ws=org_d,verbose=True)

rd = pd.DataFrame.from_records(m.sfr.reach_data).copy()
print(rd)
rd.loc[:,"flow"] = -1
rd.loc[0,"flow"] = 10000.
rd.loc[:,"flow"] = rd.flow.astype(np.float32)
rd.loc[:,"stage"] = -999.
rd.loc[:,"cond"] = 1e3
rd.loc[:,"cond"] = rd.cond.astype(np.float32)
rd.loc[:,"stage"] = rd.stage.astype(np.float32)


rd.loc[:,"segment"] = rd.ireach #iseg
rd.loc[:,"reach"] = rd.iseg
rd.loc[:,"stop"] = rd.strtop
rd.loc[:,"sbot"] = rd.strtop - 1.0

rd.loc[:,"width"] = 5.0
rd.loc[:,"width"] = rd.width.astype(np.float32)
rd.loc[:,"rough"] = 0.1
rd.loc[:,"rough"] = rd.rough.astype(np.float32)
#dt = flopy.modflow.ModflowStr.get_default_dtype()[0]
spd = {0:rd.loc[:,["k","i","j","segment","reach","flow","stage","cond","sbot","stop","width","slope","rough"]].to_records(index=False)}

m.remove_package("sfr")

flopy.modflow.ModflowStr(m,40,40,0,0,1,ipakcb=-50,istcb2=69,stress_period_data=spd,
                         const=128390,iptflg=0,)

m.remove_package("nwt")
flopy.modflow.ModflowPcg(m,hclose=0.1,rclose=100,mxiter=200,iter1=100)
m.change_model_ws(new_d,reset_external=True)
m.version = "mf2005"

m.write_input()
nam_lines = open(os.path.join(new_d,"freyberg.nam"),'r').readlines()
with open(os.path.join(new_d,"freyberg.nam"),'w') as f:
    for line in nam_lines:
        f.write(line.replace("UPW ","LPF "))
pyemu.os_utils.run("mf2005 freyberg.nam",cwd=new_d)

# hds = flopy.utils.HeadFile(os.path.join(new_d,"freyberg.hds"),model=m)
#
# hds.plot()
# plt.show()

#cb = flopy.utils.CellBudgetFile(os.path.join(new_d,"freyberg.cbc"),model=m,precision="single")
#print(cb.get_data(text="stream leakage"))

#cb = flopy.utils.CellBudgetFile(os.path.join(new_d,"freyberg.STR.cbc"),model=m,precision="single")
#print(cb.get_data(text="stream flow"))


