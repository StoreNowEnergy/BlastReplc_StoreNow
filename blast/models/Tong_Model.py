# calendar part is correct now
# which means the q1,q2,q3 part is correct
# fitted model using t_hour and T_K as inputs, (SoC is 0-1 same)
# that is the resaon why converting t_hours = t_days / 24 before passing to the model
# this conversion does not influence the trajectory function or state function(t or delta_t)


# cycle part is also correct now
# which means the q5,q6 and q7,q8,q9 part

# all other internal functions are the same as NREL(lfp_gr_SonyMurata3Ah_Battery)

import numpy as np
import scipy.stats as stats
from blast.models.degradation_model import BatteryDegradationModel
# from mylib.degradation_model import BatteryDegradationModel


class TongWu_SELECT(BatteryDegradationModel):
    """
    Formula-based degradation model (NREL-style structure).
    Calendar fade: sigmoid
    Cycle fade: break-in (sigmoid) + long-term (power law)
    """

    def __init__(self, degradation_scalar: float = 1, label: str = "Formula-based model"):
        # States
        self.states = {
            'qLoss_LLI_t': np.array([0]),
            'qLoss_LLI_EFC': np.array([0]),
            'qLoss_BreakIn_EFC': np.array([0]),
            'rGain_LLI_t': np.array([0]),
            'rGain_LLI_EFC': np.array([0]),
        }

        # Outputs
        self.outputs = {
            'q': np.array([1]),
            'q_LLI_t': np.array([1]),
            'q_LLI_EFC': np.array([1]),
            'q_BreakIn_EFC': np.array([1]),
            'r': np.array([1]),
            'r_LLI_t': np.array([1]),
            'r_LLI_EFC': np.array([1]),
        }

        # Stressors
        self.stressors = {
            'delta_t_days': np.array([np.nan]), 
            't_days': np.array([0]),
            'delta_efc': np.array([np.nan]), 
            'efc': np.array([0]),
            'TdegK': np.array([np.nan]),
            'soc': np.array([np.nan]), 
            'Ua': np.array([np.nan]), 
            'dod': np.array([np.nan]), 
            'Crate': np.array([np.nan]),
        }

        # Rates
        self.rates = {
            'q1': np.array([np.nan]),
            'q3': np.array([np.nan]),
            'q5': np.array([np.nan]),
            'q7': np.array([np.nan]),
        }

        self._degradation_scalar = degradation_scalar
        self._label = label

    # -------------------------------
    # >>> 修改部分 1: 参数表 (_params_life)
    # -------------------------------
    @property
    def _params_life(self):
        return {
            # 'q2': 0.000260239076397,   # calendar logistic slope #wrong formula fitting results
            'q2': 9.701335e-06,        # correct
            # 'q8': 0.00211968,          # break-in param
            'q8': 0.00211968,          # my fitted model break-in param    
            'q9': 0.976127,            # my fitted model break-in param
            # 'q9': 0.976127,            # break-in param
            'q6': 1.5833,              # my fitted model long-term power exponent
            # 'q6': 0.495513,            # long-term power exponent
        }

    # -------------------------------
    # >>> 修改部分 2: update_rates
    # -------------------------------
    def update_rates(self, stressors: dict):
        T_K = stressors["TdegK"]
        T_C = stressors["TdegK"] - 273.15
        SOC = stressors["soc"]       # 数组，NREL 方法
        DOD = stressors["dod"]
        Crate = stressors["Crate"]
        t_secs = stressors["t_secs"]
        delta_t_secs = t_secs[-1] - t_secs[0]

        # ---- q1 (calendar fade amplitude) ----
        # q1 = (
        #     0.03965602 * np.exp(SOC)
        #     - 0.11354792
        #     + 6.3330784 / (368.46512 - (T_C+273.15) - 7.9343367 * np.power(SOC, 1/16))
        # )

        # wrong formula fitting results
        # q1 = ((-0.67408174 + SOC) * 0.089100726) + (np.exp(np.sqrt(T_K)) * 2.1676072e-9) 
        
        # correct formula fitting results
        q1 = ((((T_K - 527.7755) - SOC) * ((T_K / 4506.6494) - 0.009831837 * SOC)) - (1.8658828 * SOC)) + 15.286845 

        # wrong formula fitting results
        # q3 = ((((T_K * -0.0066229436) + SOC) * 4028.658) * SOC) + 9237.062 

        # correct formula fitting results
        q3 = SOC + (T_K * (((0.0010451388 * SOC) + ((np.exp(SOC) * -0.00025713712) - T_K) + T_K) / -0.1327307)) + 0.26610368 
        # ---- q3 (calendar fade shift) ----
        # q3 = (
        #     9442.154
        #     - 14.3700448941758 * (T_C+273.15) * np.sqrt(SOC)
        #     - np.exp(SOC)
        #     - 2551.87 / ((T_C+273.15) - 278.5046)
        # )
        
        # ---- q7 (Break-in amplitude) ----
        # q7 = (
        #     0.096297100526572*SOC
        #     + 1.5260428*(Crate*DOD**0.5566806 - SOC + 0.36718386)
        #     * np.exp(-1.5779697*DOD/(SOC - 1/(T_C**0.4758717)))
        # )

        # my fitted model
        q7 = (
            (-0.8291941**T_C) / (DOD - 0.035703234)
            + 0.04101765*SOC*np.exp(-DOD/Crate)
            / ((1.59486547732338*DOD)**DOD + np.log(SOC))
            + 0.052813444
        )

        # ---- q5 (long-term amplitude, a_long) ----
        # q5 = (
        #     (SOC*np.log(T_C**(-0.085963376)) + 0.053993557)
        #     * (1.5287684**Crate)**Crate
        #     * np.log(SOC + (-Crate + DOD)**T_C)
        # )

        # my fitted model
        q5 = (
            DOD**1.2094425 * 8.835446e-6
            / ((SOC + 0.0649427)**Crate * SOC)
        )

        # ---- 按 NREL 方法取平均 ----
        q1 = np.trapz(q1, x=t_secs) / delta_t_secs
        q3 = np.trapz(q3, x=t_secs) / delta_t_secs
        q5 = np.trapz(q5, x=t_secs) / delta_t_secs # no time varying inputs
        # NREL q5 不随时间变，不积分
        q7 = np.trapz(q7, x=t_secs) / delta_t_secs
        

        # 存储结果
        # rates = {"q1": q1, "q3": q3, "q5": q5, "q7": q7}
        rates = np.array([q1, q3, q5, q7,])
        for k, v in zip(self.rates.keys(), rates):
            self.rates[k] = np.append(self.rates[k], v)

    # -------------------------------
    # NREL 原有: update_states
    # -------------------------------
    def update_states(self, stressors: dict):
        delta_t_days = stressors["delta_t_days"]
        delta_efc = stressors["delta_efc"]

        p = self._params_life

        # 最新速率
        r = self.rates.copy()
        for k, v in zip(r.keys(), r.values()):
            r[k] = v[-1]

        # Calendar fade (sigmoid)
        dq_LLI_t = self._degradation_scalar * self._update_sigmoid_state(
            self.states['qLoss_LLI_t'][-1],
            delta_t_days*24,  # days -> hours
            r['q1'], p['q2'], r['q3']
        )

        # if delta_efc / delta_t_days > 3: # only evalaute if more than 3 full cycles per day
        #             dq_BreakIn_EFC = self._degradation_scalar * self._update_sigmoid_state(states['qLoss_BreakIn_EFC'][-1], delta_efc, r['q7'], p['q8'], p['q9'])
        # else:
        #             dq_BreakIn_EFC = 0

        # Break-in fade (sigmoid)
        if delta_efc / delta_t_days > 3: # only evalaute if more than 3 full cycles per day
            dq_BreakIn_EFC = self._degradation_scalar * self._update_sigmoid_state(
                self.states['qLoss_BreakIn_EFC'][-1],
                delta_efc,
                r['q7'], p['q8'], p['q9']
            )
        else:
            dq_BreakIn_EFC = 0
        
        

        # Long-term cycle fade (power law B)
        dq_LLI_EFC = self._degradation_scalar * self._update_power_B_state(
            self.states['qLoss_LLI_EFC'][-1],
            delta_efc,
            r['q5'], p['q6']
        )

        # 更新 states
        dx = [dq_LLI_t, dq_LLI_EFC, dq_BreakIn_EFC, 0, 0]
        for k, v in zip(self.states.keys(), dx):
            x = self.states[k][-1] + v
            self.states[k] = np.append(self.states[k], x)

    # -------------------------------
    # NREL 原有: update_outputs
    # -------------------------------
    def update_outputs(self, stressors):
        states = self.states

        # Capacity
        q_LLI_t = 1 - states['qLoss_LLI_t'][-1]
        q_LLI_EFC = 1 - states['qLoss_LLI_EFC'][-1]
        q_BreakIn_EFC = 1 - states['qLoss_BreakIn_EFC'][-1]
        q = 1 - states['qLoss_LLI_t'][-1] - states['qLoss_LLI_EFC'][-1] - states['qLoss_BreakIn_EFC'][-1]

        # if delta_efc / delta_t_days > 3: # only evalaute if more than 3 full cycles per day
        #             dq_BreakIn_EFC = self._degradation_scalar * self._update_sigmoid_state(states['qLoss_BreakIn_EFC'][-1], delta_efc, r['q7'], p['q8'], p['q9'])
        # else:
        #             dq_BreakIn_EFC = 0


        # Resistance (not used in your formulas, keep as-is)
        r_LLI_t = 1 + states['rGain_LLI_t'][-1]
        r_LLI_EFC = 1 + states['rGain_LLI_EFC'][-1]
        r = 1 + states['rGain_LLI_t'][-1] + states['rGain_LLI_EFC'][-1]

        # Assemble output
        out = np.array([q, q_LLI_t, q_LLI_EFC, q_BreakIn_EFC, r, r_LLI_t, r_LLI_EFC])
        for k, v in zip(self.outputs.keys(), out):
            self.outputs[k] = np.append(self.outputs[k], v)
