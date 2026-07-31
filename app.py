import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from scipy.optimize import minimize_scalar
from matplotlib import rcParams

# إعدادات الصفحة
st.set_page_config(page_title="ThermoTwin", page_icon="️", layout="wide")

# ==========================================
# محرك خصائص الموائع
# ==========================================

class LibyanCrudeProperties:
    def __init__(self, api_gravity=35):
        self.api = api_gravity
        self.rho_15C = 141.5 * 1000 / (api_gravity + 131.5)
        
    def get_density(self, T_c):
        return self.rho_15C - 0.65 * (T_c - 15.0)
    
    def get_cp(self, T_c):
        return 1800.0 + 2.5 * T_c
    
    def get_viscosity(self, T_c):
        T_k = T_c + 273.15
        return 2.8e-6 * np.exp(2600.0 / T_k)
    
    def get_thermal_conductivity(self, T_c):
        return 0.13 - 0.0001 * T_c


class UtilityFluidProperties:
    def __init__(self, fluid_name="Water"):
        self.fluid = fluid_name
    
    def get_properties(self, T_c, P_pa=1000000):
        T_k = T_c + 273.15
        try:
            rho = CP.PropsSI('D', 'T', T_k, 'P', P_pa, self.fluid)
            cp = CP.PropsSI('C', 'T', T_k, 'P', P_pa, self.fluid)
            mu = CP.PropsSI('V', 'T', T_k, 'P', P_pa, self.fluid)
            k = CP.PropsSI('L', 'T', T_k, 'P', P_pa, self.fluid)
            return rho, cp, mu, k
        except:
            return 850, 2200, 0.002, 0.12


class AdvancedHeatExchanger:
    def __init__(self, Area, tube_ID, tube_OD, num_tubes, tube_length, tube_passes=2):
        self.Area = Area
        self.tube_ID = tube_ID
        self.tube_OD = tube_OD
        self.num_tubes = num_tubes
        self.tube_length = tube_length
        self.tube_passes = tube_passes
        tubes_per_pass = num_tubes / tube_passes
        self.flow_area_tube = tubes_per_pass * (np.pi * (tube_ID**2) / 4)
    
    def calculate_h_tube_side(self, m_flow, crude_props, T_in, T_out):
        T_bulk = (T_in + T_out) / 2.0
        rho = crude_props.get_density(T_bulk)
        cp = crude_props.get_cp(T_bulk)
        mu = crude_props.get_viscosity(T_bulk)
        k = crude_props.get_thermal_conductivity(T_bulk)
        
        velocity = m_flow / (rho * self.flow_area_tube)
        Re = (rho * velocity * self.tube_ID) / mu
        Pr = (mu * cp) / k
        
        n = 0.4 if T_out > T_in else 0.3
        Nu = 0.023 * (Re**0.8) * (Pr**n)
        h_i = (Nu * k) / self.tube_ID
        
        return h_i, Re, Pr, rho, cp, mu, k
    
    def calculate_h_shell_side(self, m_flow, utility_props, T_in, T_out):
        T_bulk = (T_in + T_out) / 2.0
        rho, cp, mu, k = utility_props.get_properties(T_bulk)
        velocity = 0.5
        Re = (rho * velocity * self.tube_OD) / mu
        Pr = (mu * cp) / k
        Nu = 0.36 * (Re**0.55) * (Pr**(1/3))
        h_o = (Nu * k) / self.tube_OD
        return h_o
    
    def calculate_performance(self, m_h, m_c, Th_in, Tc_in, Th_out, Tc_out):
        crude = LibyanCrudeProperties(api_gravity=35)
        utility = UtilityFluidProperties("Water")
        
        h_i, Re_h, Pr_h, rho_h, cp_h, mu_h, k_h = self.calculate_h_tube_side(m_h, crude, Th_in, Th_out)
        h_o = self.calculate_h_shell_side(m_c, utility, Tc_in, Tc_out)
        
        R_wall = 0.0001
        U_clean_dynamic = 1.0 / (1.0/h_i + 1.0/h_o + R_wall)
        
        dT1 = Tc_in - Th_out
        dT2 = Tc_out - Th_in
        
        if dT1 <= 0 or dT2 <= 0 or abs(dT1 - dT2) < 1e-5:
            LMTD = abs(dT1 + dT2) / 2
        else:
            LMTD = (dT1 - dT2) / np.log(dT1 / dT2)
        
        Q_h = m_h * cp_h * abs(Th_in - Th_out)
        rho_c, cp_c, mu_c, k_c = utility.get_properties((Tc_in + Tc_out)/2)
        Q_c = m_c * cp_c * abs(Tc_out - Tc_in)
        Q_actual = (Q_h + Q_c) / 2.0
        
        U_actual = Q_actual / (self.Area * LMTD)
        Rf = max(0.0, (1.0 / U_actual) - (1.0 / U_clean_dynamic))
        
        return {
            'U_clean': U_clean_dynamic,
            'U_actual': U_actual,
            'Rf': Rf,
            'h_i': h_i,
            'h_o': h_o,
            'Re': Re_h,
            'Pr': Pr_h,
            'Q': Q_actual,
            'LMTD': LMTD,
            'mu': mu_h,
            'rho': rho_h
        }
    
    def calculate_outlet_temperatures(self, m_h, m_c, Th_in, Tc_in, U_actual):
        crude = LibyanCrudeProperties(api_gravity=35)
        utility = UtilityFluidProperties("Water")
        
        cp_h = crude.get_cp((Th_in + 100) / 2)
        rho_c, cp_c, mu_c, k_c = utility.get_properties((Tc_in + 80) / 2)
        
        C_h = m_h * cp_h
        C_c = m_c * cp_c
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        C_r = C_min / C_max
        
        NTU = (U_actual * self.Area) / C_min
        
        if C_r < 1.0:
            effectiveness = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
        else:
            effectiveness = NTU / (1 + NTU)
        
        Q_max = C_min * (Tc_in - Th_in)
        Q_actual = effectiveness * Q_max
        
        Th_out = Th_in + Q_actual / C_h
        Tc_out = Tc_in - Q_actual / C_c
        
        return Th_out, Tc_out, Q_actual


# ==========================================
# واجهة Streamlit
# ==========================================

st.title("️ ThermoTwin: التوأم الرقمي الفيزيائي للمبادلات الحرارية")
st.markdown("""
نظام محاكاة متقدم يحسب خصائص الموائع ديناميكياً ويستخدم معادلات الانتقال الحراري المعتمدة
لحساب الأداء الفعلي ومقاومة التلوث وتحسين التكلفة الاقتصادية للصيانة.
""")

# الشريط الجانبي للمدخلات
st.sidebar.header("⚙️ أبعاد المبادل")
Area = st.sidebar.number_input("المساحة (m²)", value=150.0, min_value=50.0, max_value=300.0)
tube_OD = st.sidebar.number_input("القطر الخارجي (mm)", value=19.05, min_value=15.0, max_value=25.0)
num_tubes = st.sidebar.number_input("عدد الأنابيب", value=150, min_value=50, max_value=400)
tube_ID = st.sidebar.number_input("القطر الداخلي (mm)", value=16.6, min_value=12.0, max_value=20.0)
tube_length = st.sidebar.number_input("طول الأنبوب (m)", value=4.88, min_value=3.0, max_value=6.0)
tube_passes = st.sidebar.selectbox("عدد الممرات", [1, 2, 4, 6], index=1)

st.sidebar.header("🛢️ ظروف التشغيل")
m_h = st.sidebar.slider("تدفق الخام (kg/s)", 10.0, 80.0, 30.0)
m_c = st.sidebar.slider("تدفق المائع الساخن (kg/s)", 15.0, 100.0, 35.0)
Th_in = st.sidebar.slider("حرارة الخام الداخلة (°C)", 20.0, 60.0, 30.0)
Tc_in = st.sidebar.slider("حرارة المائع الساخن (°C)", 120.0, 200.0, 150.0)

st.sidebar.header("💰 المعلمات الاقتصادية")
Fuel_cost = st.sidebar.number_input("تكلفة الوقود ($/J)", value=3e-9, format="%.1e")
Cleaning_cost = st.sidebar.number_input("تكلفة التنظيف ($)", value=15000.0)
Downtime_cost = st.sidebar.number_input("تكلفة التوقف ($/يوم)", value=4000.0)

# ==========================================
# المحاكاة
# ==========================================

@st.cache_data
def run_simulation(Area, tube_ID, tube_OD, num_tubes, tube_length, tube_passes, m_h, m_c, Th_in, Tc_in):
    tube_OD_m = tube_OD / 1000
    tube_ID_m = tube_ID / 1000
    
    hx = AdvancedHeatExchanger(Area, tube_ID_m, tube_OD_m, num_tubes, tube_length, tube_passes)
    
    days = np.linspace(0, 120, 600)
    Rf_inf = 0.0008
    k_fouling = 0.04
    Rf_simulated = Rf_inf * (1 - np.exp(-k_fouling * days))
    
    results = []
    for i, day in enumerate(days):
        m_h_var = m_h + np.random.normal(0, 0.5)
        m_c_var = m_c + np.random.normal(0, 0.5)
        Th_in_var = Th_in + np.random.normal(0, 0.5)
        Tc_in_var = Tc_in + np.random.normal(0, 0.5)
        
        crude = LibyanCrudeProperties(api_gravity=35)
        utility = UtilityFluidProperties("Water")
        
        h_i, Re_h, Pr_h, rho_h, cp_h, mu_h, k_h = hx.calculate_h_tube_side(m_h_var, crude, Th_in_var, 100)
        h_o = hx.calculate_h_shell_side(m_c_var, utility, Tc_in_var, 80)
        
        R_wall = 0.0001
        U_clean = 1.0 / (1.0/h_i + 1.0/h_o + R_wall)
        U_actual = 1.0 / (1.0/U_clean + Rf_simulated[i])
        
        Th_out, Tc_out, Q_actual = hx.calculate_outlet_temperatures(
            m_h_var, m_c_var, Th_in_var, Tc_in_var, U_actual
        )
        
        Th_out += np.random.normal(0, 0.3)
        Tc_out += np.random.normal(0, 0.3)
        
        perf = hx.calculate_performance(m_h_var, m_c_var, Th_in_var, Tc_in_var, Th_out, Tc_out)
        
        results.append({
            'Day': day,
            'Rf_calc': perf['Rf'],
            'Rf_sim': Rf_simulated[i],
            'U_actual': perf['U_actual'],
            'U_clean': perf['U_clean'],
            'h_i': perf['h_i'],
            'h_o': perf['h_o'],
            'Re': perf['Re'],
            'Pr': perf['Pr'],
            'Q': perf['Q'],
            'mu': perf['mu'],
            'rho': perf['rho'],
            'Th_out': Th_out,
            'Tc_out': Tc_out
        })
    
    return pd.DataFrame(results)

df = run_simulation(Area, tube_ID, tube_OD, num_tubes, tube_length, tube_passes, m_h, m_c, Th_in, Tc_in)

# ==========================================
# عرض النتائج
# ==========================================

st.markdown("###
