import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import os

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray as xr
from glob import glob


filei = 'params/params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
with open(filei, 'rb') as f:
	ckpt = checkpoint.load(f, graphcast.CheckPoint)

params = ckpt.params
state = {}
model_config = ckpt.model_config

task_config = ckpt.task_config

with open("stats/stats_diffs_stddev_by_level.nc","rb") as f:
  diffs_stddev_by_level = xr.load_dataset(f).compute()
with open("stats/stats_mean_by_level.nc","rb") as f:
  mean_by_level = xr.load_dataset(f).compute()
with open("stats/stats_stddev_by_level.nc","rb") as f:
  stddev_by_level = xr.load_dataset(f).compute()


# @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)
  
  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)
  
  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)
      
  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


files = sorted(glob('/scratch/08589/hvtran/GC/input/*.nc'))

filei = files[0]
for filei in files[1:]:
	ds =xr.load_dataset(filei).compute() 
	ds1 = ds.drop_dims('level')
	ds1 = ds1.rename_dims({'levels':'level'})
	ds1 = ds1.assign_coords(level=ds.level)
	ds1 = ds1.drop_vars('toa_incident_solar_radiation')
	
	for i in range(4):
		eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
			ds1, target_lead_times="0h",
			**dataclasses.asdict(task_config))
		
		print("Inputs:  ", eval_inputs.dims.mapping)
		print("Targets: ", eval_targets.dims.mapping)
		print("Forcings:", eval_forcings.dims.mapping)
		
		eval_targets = eval_targets.assign_coords(time = eval_targets.time+21600000000000)
		eval_forcings = eval_forcings.assign_coords(time = eval_forcings.time+21600000000000)
		
		for vari in list(eval_inputs.variables):
			if eval_inputs[vari].data.dtype ==np.float64:
				#print(vari)
				eval_inputs[vari].data = eval_inputs[vari].data.astype(np.float32)
		
		for vari in list(eval_targets.variables):
			if eval_targets[vari].data.dtype ==np.float64:
				#print(vari)
				eval_targets[vari].data = eval_targets[vari].data.astype(np.float32)
		
		for vari in list(eval_forcings.variables):
			if eval_forcings[vari].data.dtype ==np.float64:
				#print(vari)
				eval_forcings[vari].data = eval_forcings[vari].data.astype(np.float32)
		
		predictions = rollout.chunked_prediction(
			run_forward_jitted,
			rng=jax.random.PRNGKey(0),
			inputs=eval_inputs,
			targets_template=eval_targets * np.nan,
			forcings=eval_forcings)
		
		out_file = os.path.basename(filei).split('.')[0] + '_'+str(i)+'.nc'
		print(out_file)
		predictions.to_netcdf('/scratch/08589/hvtran/GC/output/'+out_file)
		####Append new prediction to ds
			
		t0 = ds1.datetime[1]
		t1 = t0+21600000000000
		
		
		new_ds = xr.concat([ds1.isel(time=1), predictions], dim='time')
		new_ds = new_ds.assign_coords(time=np.array([0, 21600000000000], dtype='timedelta64[ns]'))
		new_ds = new_ds.assign_coords(datetime = np.array([t0.values,t1.values]))
		new_ds = new_ds.transpose("batch","time","level","lat","lon","datetime")
		
		new_ds['geopotential_at_surface'] = new_ds['geopotential_at_surface'].isel(time=0).drop_vars('time')
		new_ds['land_sea_mask'] = new_ds['land_sea_mask'].isel(time=0).drop_vars('time')
		
		ds1 = new_ds.copy()

