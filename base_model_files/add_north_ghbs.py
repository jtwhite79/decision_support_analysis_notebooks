import flopy

m = flopy.modflow.Modflow.load("freyberg.nam",check=False)
#m.change_model_ws("bak")
#m.write_input()
ib = m.bas6.ibound.array
ghb_data = [list(d) for d in list(m.ghb.stress_period_data[0])]
for k in range(m.nlay):
	for j in range(m.ncol):
		if ib[k,0,j] <= 0:
			continue
		ghb_data.append([k,0,j,34.0,1000.0])
print(ghb_data)
_ = flopy.modflow.ModflowGhb(m,stress_period_data=ghb_data,ipakcb=m.ghb.ipakcb)
#m.write_input()
#m.change_model_ws("temp")
m.write_input()