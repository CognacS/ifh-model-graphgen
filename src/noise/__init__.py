import torch.nn as nn

class TimeSampler:

	def sample_time(self, **kwargs):
		raise NotImplementedError


class NoiseSchedule (nn.Module):

	def __init__(self):
		super().__init__()

	def params_next(self, t, **kwargs):
		raise NotImplementedError
	def params_time_t(self, t, **kwargs):
		raise NotImplementedError
	def params_posterior(self, t, **kwargs):
		raise NotImplementedError
	def sample_time(self, **kwargs):
		raise NotImplementedError
	def get_time_from_param(self, param, **kwargs):
		raise NotImplementedError
	def get_max_time(self, **kwargs):
		raise NotImplementedError
	def reverse_step(self, t, **kwargs):
		raise NotImplementedError
	def prepare_data(self, datapoint, **kwargs):
		pass


class NoiseProcess (nn.Module):

	def __init__(self, schedule : NoiseSchedule):
		"""
		Parameters
		----------
		schedule : DiffusionSchedule
			gives the parameter values for next, sample_t, posterior
		"""
		super().__init__()
		self.schedule = schedule

	############################################################################
	#                                 TO DEFINE                                #
	############################################################################

	# sample datapoint from the stationary distribution (t->+inf)
	def sample_stationary(self, **kwargs):
		raise NotImplementedError

	# sample noise
	def sample_noise_next(self, current_datapoint, t, **kwargs):
		raise NotImplementedError

	def sample_noise_from_original(self, original_datapoint, t, **kwargs):
		raise NotImplementedError
	
	def sample_noise_posterior(self, original_datapoint, current_datapoint, t, **kwargs):
		raise NotImplementedError

	# apply noise
	def apply_noise_next(self, current_datapoint, noise, t, **kwargs):
		raise NotImplementedError

	def apply_noise_from_original(self, original_datapoint, noise, t, **kwargs):
		raise NotImplementedError
	
	def apply_noise_posterior(self, original_datapoint, current_datapoint, noise, t, **kwargs):
		raise NotImplementedError

	############################################################################
	#                              ALREADY DEFINED                             #
	############################################################################


	###############################  PARAMETERS  ###############################
	def get_schedule(self):
		return self.schedule

	def get_params_next(self, t, **kwargs):
		return self.schedule.params_next(t, **kwargs)

	def get_params_from_original(self, t, **kwargs):
		return self.schedule.params_time_t(t, **kwargs)

	def get_params_posterior(self, t, **kwargs):
		return self.schedule.params_posterior(t, **kwargs)

	def get_max_time(self, **kwargs):
		return self.schedule.get_max_time(**kwargs)
	
	def reverse_time(self, t, **kwargs):
		return self.schedule.reverse_step(t, **kwargs)
	
	def normalize_time(self, t, **kwargs):
		max_time = self.get_max_time(normalize=True, **kwargs)
		if max_time is None:
			return t
		else:
			return t / max_time
		
	def normalize_reverse_time(self, t, **kwargs):
		rev_time = self.reverse_time(t, **kwargs)
		return self.normalize_time(rev_time, **kwargs)
	

	def prepare_data(self, datapoint, **kwargs):
		self.schedule.prepare_data(datapoint, **kwargs)


	############################  SAMPLING METHODS  ############################

	def sample_next(self, current_datapoint, t, noise=None, **kwargs):
		
		if noise is None:
			noise = self.sample_noise_next(current_datapoint, t, **kwargs)

		next_datapoint = self.apply_noise_next(current_datapoint, noise, t, **kwargs)

		return next_datapoint
		

	def sample_from_original(self, original_datapoint, t, noise=None, **kwargs):
				
		if noise is None:
			noise = self.sample_noise_from_original(original_datapoint, t, **kwargs)

		step_t_datapoint = self.apply_noise_from_original(original_datapoint, noise, t, **kwargs)

		return step_t_datapoint
	

	def sample_posterior(self, original_datapoint, current_datapoint, t, noise=None, **kwargs):
				
		if noise is None:
			noise = self.sample_noise_posterior(original_datapoint, current_datapoint, t, **kwargs)

		prev_datapoint = self.apply_noise_posterior(original_datapoint, current_datapoint, noise, t, **kwargs)

		return prev_datapoint