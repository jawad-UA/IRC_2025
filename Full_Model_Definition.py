import torch
import torch.nn as nn
import sys
import numpy as np
import torch
import torch.nn.functional as F

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from Functions_KGE_Direct_Arrays import kge_loss, get_direct_arrays, get_direct_arrays_scaled, canopy_NN, Rs_soil, gkge_static, compute_gkge, GKGE_v1
from Functions_KGE_Direct_Arrays import kge_loss, get_direct_arrays, get_direct_arrays_scaled, canopy_NN, Rs_soil,ET_surrogate, SimpleMonotonicETSurrogate,gkge_static_median_rearranged, compute_gkge_median_rearranged, split_train_eval_by_year_extremes, extract_Qs_train

np.random.seed(4000)
torch.autograd.set_detect_anomaly(False)
#torch.manual_seed(150)



class MA3_PM(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_ET_surrogate=ET_surrogate()
        self.C_ET_surrogate=ET_surrogate()
        self.S_ET_surrogate=ET_surrogate()
        self.B_ET_surrogate=ET_surrogate()
        
        
        ##===Aerodynamic Resistance Canopy=====
        self.C_ra_a_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_ra_b_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_ra_a_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_ra_b_theta=nn.Parameter(torch.randn(1) * 0.1)
        ###=======F_veg=====Gate
        self.b_f_veg=nn.Parameter(torch.randn(1) * 0.1)
        self.a_f_veg=nn.Parameter(torch.randn(1) * 0.1)
        
        self.precip_split_param_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.precip_split_param_theta_2=nn.Parameter(torch.randn(1) * 0.1)
        
        self.G_a1_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_b1_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_b2_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_b3_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_b4_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_b5_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_b6_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_max_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_max_b_parameter_theta=nn.Parameter(torch.randn(1) * 0.1)
        ##=====Infiltration Gate Parameters===###
        self.G_P_max_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.G_P_max_b_parameter_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        #######  Canopy Bucket=======
        ##=====Overflow Gate Parameters===###
        self.C_max_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_max_b_parameter_theta=nn.Parameter(torch.randn(1) * 0.1)
        ##=====Infiltration Gate Parameters===###
        self.C_P_max_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_P_max_b_parameter_theta=nn.Parameter(torch.randn(1) * 0.1)
        ##=====Loss Gate Parameters
        self.C_b1_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_b2_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_b3_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_b4_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_b5_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.C_b6_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        self.C_a1_theta=nn.Parameter(torch.randn(1) * 0.1)
        ##====Max rate ET parameters=====###
        self.C_c_ET=nn.Parameter(torch.randn(1) * 0.1)
        self.C_c_seepage=nn.Parameter(torch.randn(1) * 0.1)
        self.C_c_out=nn.Parameter(torch.randn(1) * 0.1)
        self.C_c_remember=nn.Parameter(torch.randn(1) * 0.1)
        self.C_b_seepage_X=nn.Parameter(torch.randn(1) * 0.1)
        self.C_a_seepage=nn.Parameter(torch.randn(1) * 0.1)
        self.C_b_out_X=nn.Parameter(torch.randn(1) * 0.1)
        self.C_a_out=nn.Parameter(torch.randn(1) * 0.1)
        #######  Soil Bucket=======
        ##====P_excesss====Parameters
        self.P_in_excess_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        ##=====Loss Gate Parameters=====###
        self.S_b1_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_b2_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_b3_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_b4_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_b5_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_b6_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        self.S_a1_theta=nn.Parameter(torch.randn(1) * 0.1)
        ##=====Overflow Gate Parameters===###
        self.S_max_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_max_b_parameter_theta=nn.Parameter(torch.randn(1) * 0.1)
        ##====Infiltration Excess=====###
        self.S_P_max_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.S_P_max_b_parameter_theta=nn.Parameter(torch.randn(1) * 0.1)
        ##=====Seepage Gate Parameters===###
        self.S_b_seepage_X=nn.Parameter(torch.randn(1) * 0.1)
        self.S_a_seepage=nn.Parameter(torch.randn(1) * 0.1)
        self.S_c_seepage=nn.Parameter(torch.randn(1) * 0.1)
        
        ##=====Out Gate Parameters===###
        self.S_b_out_X=nn.Parameter(torch.randn(1) * 0.1)
        self.S_a_out=nn.Parameter(torch.randn(1) * 0.1)
        self.S_c_out=nn.Parameter(torch.randn(1) * 0.1)
        self.S_c_ET=nn.Parameter(torch.randn(1) * 0.1)
        ##=====Remember Gate Parameters===###
        self.S_c_remember=nn.Parameter(torch.randn(1) * 0.1)
        
        
        #######  Baseflow Bucket=======
        ##=====Loss Gate Parameters=====###
        self.B_b1_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.B_b2_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.B_b3_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.B_b4_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.B_b5_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.B_b6_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        self.B_a1_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        ##=====Out Gate Parameters===###
        self.B_b_out_X=nn.Parameter(torch.randn(1) * 0.1)
        self.B_a_out=nn.Parameter(torch.randn(1) * 0.1)
        ##=====Max Rate Parameters====##
        self.B_c_out=nn.Parameter(torch.randn(1) * 0.1)
        self.B_c_ET=nn.Parameter(torch.randn(1) * 0.1)
        self.B_c_remember=nn.Parameter(torch.randn(1) * 0.1)
        
        #######  Delay Bucket=======
        ##=====Out Gate Parameters===###
        self.D_b_out_X=nn.Parameter(torch.randn(1) * 0.1)
        self.D_a_out=nn.Parameter(torch.randn(1) * 0.1)
        
        ###=====Snow Related Gates====####
        self.b1_melting_canopy_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.b2_melting_canopy_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.c_melting_canopy_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        self.b1_melting_soil_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.b2_melting_soil_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.c_melting_soil_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        self.b_rain_fraction_theta=nn.Parameter(torch.randn(1) * 0.1)
        self.c_rain_fraction_theta=nn.Parameter(torch.randn(1) * 0.1)
        
        
    def forward(self, forcing_data, scaled_forcing_data,P_scale, T_scale, K):
        Rn, LAI, Deficit, lamda, delt, gama, um=get_direct_arrays(forcing_data)
        Rn_scaled, LAI_scaled, Deficit_scaled, lamda_scaled, delt_scaled, gama_scaled, um_scaled=get_direct_arrays_scaled(forcing_data)
        
        
        b_f_veg1=torch.exp(self.b_f_veg)+1
        
        
        G_max=torch.exp(self.G_max_theta) #torch.exp(self.C_max_theta)#torch.clamp(torch.exp(self.C_max_theta)/2, max=2)
        G_max_b_parameter = torch.clamp(0.5 + torch.exp(self.G_max_b_parameter_theta), max=5.0) #0.5+torch.exp(self.C_max_b_parameter_theta)
        G_P_max=torch.exp(self.G_P_max_theta)#(torch.sigmoid(self.C_max_theta/10) * (10 - 0.1) + 1)#torch.exp(self.C_max_theta) #torch.exp(self.C_max_theta)#torch.clamp(torch.exp(self.C_max_theta)/2, max=2)
        G_P_max_b_parameter=torch.clamp(0.5 + torch.exp(self.G_P_max_b_parameter_theta), max=5.0)

        
        C_max=torch.exp(self.C_max_theta) #torch.exp(self.C_max_theta)#torch.clamp(torch.exp(self.C_max_theta)/2, max=2)
        C_max_b_parameter = torch.clamp(0.5 + torch.exp(self.C_max_b_parameter_theta), max=5.0) #0.5+torch.exp(self.C_max_b_parameter_theta)
        S_max=torch.exp(self.S_max_theta)*K
        S_max_b_parameter= torch.clamp(0.5 + torch.exp(self.S_max_b_parameter_theta), max=5.0)#0.5+torch.exp(self.S_max_b_parameter_theta)
        
        C_P_max=torch.exp(self.C_P_max_theta)#(torch.sigmoid(self.C_max_theta/10) * (10 - 0.1) + 1)#torch.exp(self.C_max_theta) #torch.exp(self.C_max_theta)#torch.clamp(torch.exp(self.C_max_theta)/2, max=2)
        C_P_max_b_parameter=torch.clamp(0.5 + torch.exp(self.C_P_max_b_parameter_theta), max=5.0)
        S_P_max=torch.exp(self.S_P_max_theta)
        S_P_max_b_parameter=torch.clamp(0.5 + torch.exp(self.S_P_max_b_parameter_theta), max=5.0)
        
        ##==========PM equation related resistance parameters=====
        C_ra_a_param=F.softplus(self.C_ra_a_theta)
        S_ra_a_param=F.softplus(self.S_ra_a_theta)
        
        
        
        G_b1=self.G_b1_theta; G_b2=self.G_b2_theta;  G_b3=self.G_b3_theta; G_b4=self.G_b4_theta;  G_b5=self.G_b5_theta; G_b6=self.G_b6_theta;
        C_b1=self.C_b1_theta; C_b2=self.C_b2_theta;  C_b3=self.C_b3_theta; C_b4=self.C_b4_theta;  C_b5=self.C_b5_theta; C_b6=self.C_b6_theta;
        S_b1=self.S_b1_theta; S_b2=self.S_b2_theta;  S_b3=self.S_b3_theta; S_b4=self.S_b4_theta;  S_b5=self.S_b5_theta; S_b6=self.S_b6_theta;
        B_b1=self.B_b1_theta; B_b2=self.B_b2_theta;  B_b3=self.B_b3_theta; B_b4=self.B_b4_theta;  B_b5=self.B_b5_theta; B_b6=self.B_b6_theta;

        P_in_excess=torch.exp(self.P_in_excess_theta)
        
        
        omega_C= torch.exp(self.C_c_ET) +torch.exp(self.C_c_out) + torch.exp(self.C_c_seepage) +torch.exp(self.C_c_remember)
        omega_C_O=torch.exp(self.C_c_out)/omega_C ; omega_C_S=torch.exp(self.C_c_seepage)/omega_C ; omega_C_R=torch.exp(self.C_c_remember)/omega_C 
        omega_C_ET=torch.exp(self.C_c_ET)/omega_C
        
        
        #Max rate of fluxes from soil bucket
        omega_S= torch.exp(self.S_c_ET) +torch.exp(self.S_c_out) + torch.exp(self.S_c_seepage) +torch.exp(self.S_c_remember)
        omega_S_O=torch.exp(self.S_c_out)/omega_S ; omega_S_S=torch.exp(self.S_c_seepage)/omega_S ; omega_S_R=torch.exp(self.S_c_remember)/omega_S 
        omega_S_ET=torch.exp(self.S_c_ET)/omega_S
        
        #Max rate of fluxes from baseflow bucket
        omega_B= torch.exp(self.B_c_ET) +torch.exp(self.B_c_out)  +torch.exp(self.B_c_remember)
        omega_B_O=torch.exp(self.B_c_out)/omega_B ; omega_B_R=torch.exp(self.B_c_remember)/omega_B 
        omega_B_ET=torch.exp(self.B_c_ET)/omega_B
        
        Precip = forcing_data[:, 3]*P_scale; T_air=forcing_data[:, 1]*T_scale;SW=scaled_forcing_data[:, 5]; T_air_scaled=scaled_forcing_data[:, 1]*T_scale; T_skin=scaled_forcing_data[:, 9]*T_scale;
        C_level=[torch.tensor([0.0])];S_level=[torch.tensor([0.0])];B_level=[torch.tensor([0.0])]; 
        G_level=[torch.tensor([0.0])]; G1_level=[torch.tensor([0.0])]; S1_level=[torch.tensor([0.0])]; 
        
        XG1=[]; XS1=[];
        ra_C=[];ra_S=[]; rs_C=[]; rs_S=[]; 
        L_C=[]; L_S=[]; L_B=[]
        OF_C=[]; OF_S=[]; OF_G=[]; L_G=[]; Gmax_dynamic=[]; G_exces=[]; OF_G=[]
        O_S=[]; O_B=[]; O_D=[]; Seepage_S=[]; Seepage_C=[]; O_C=[]
        S_exces=[]; C_exces=[];
        f_veg=[]; f_wet_G=[]; f_wet_S=[]; P_G=[]; 
        P=[]; PC=[]; GCE=[]; IC=[]; XC=[]; GCO=[]; GCL=[]; LC=[]; PS=[]; CO=[]; EC=[]; GSE=[]; ES=[]; IS=[]; LS=[]; GSL=[]; XS=[]; GSO1=[]; O1S=[]; GSO2=[]; O2S=[]; GSO3=[]; O3S=[]; XD=[]; GDO=[]; OD=[]; XB=[]; GBL=[]; LB=[]; GBO=[]; OB=[]; O=[]
        f_rain=[]; f_snow=[];  f_melt_canopy=[];f_melt_soil=[];water_melted_G1=[]; OF_G1=[];water_melted_S1=[]; OF_S1=[];
        eps=1e-08; WB_G=[]
        #print(C_max, S_max)
        #print(F.softplus(self.S_b_out_X), self.S_a_out, omega_S_S)
        #print(S_P_max, S_P_max_b_parameter )
        #print(b_f_veg1, self.a_f_veg)
        #print(omega_B_O, F.softplus(self.S_b_out_X),self.S_a_out)
        #print(torch.sigmoid(self.a_f_veg))
        #print(F.softplus(self.b_rain_fraction_theta),self.c_rain_fraction_theta )
        for i in range(len(Precip) - 1):
            
            ##==Rain fraction and Melt fraction====####
            f_rainn=torch.sigmoid(F.softplus(self.b_rain_fraction_theta)*(T_air[i]-273.15)-self.c_rain_fraction_theta)
           
            f_rain.append(f_rainn)
            f_snow.append(1-f_rain[i])
            
            f_meltt_canopy=torch.sigmoid(F.softplus(self.b1_melting_canopy_theta)*(T_skin[i])+F.softplus(self.b2_melting_canopy_theta)*SW[i]-self.c_melting_canopy_theta)
            f_melt_canopy.append(f_meltt_canopy)
            
            
            f_meltt_soil=torch.sigmoid(F.softplus(self.b1_melting_soil_theta)*(T_skin[i])+F.softplus(self.b2_melting_soil_theta)*SW[i]-self.c_melting_soil_theta)
            f_melt_soil.append(f_meltt_soil)
            
            ####++=======Canopy Bucket++==============###
            #==== Caluclating Vegetation Fraction
            scaling_param_LAI=torch.sigmoid(self.a_f_veg)
            St_f_veg=1-torch.exp(-0.1*(1.1*LAI[i]))#torch.sigmoid(b_f_veg1*torch.log1p(LAI[i])+self.a_f_veg)
            
            f_veg.append(St_f_veg)
            Gmax_dynamic.append(G_max*f_veg[i])
            
  
            ##===Calculating Precip Excess as overflow======
            water_going_to_G1=Precip[i]*f_veg[i]*f_snow[i]
            water_melted_G1.append(G1_level[i]*f_melt_canopy[i])
            water_balance_in_G1=G1_level[i]+water_going_to_G1-water_melted_G1[i]
            G1_Overflow =G_max_b_parameter*F.softplus((water_balance_in_G1-Gmax_dynamic[i])/G_max_b_parameter)  - G_max_b_parameter*F.softplus((-Gmax_dynamic[i])/G_max_b_parameter)
            
            OF_G1.append(G1_Overflow)
            G1_level.append(water_balance_in_G1-OF_G1[i])
            
            water_going_to_G=Precip[i]*f_veg[i]*f_rain[i]
            P_G.append(water_going_to_G)
            G_P_excess=G_P_max_b_parameter*F.softplus((water_going_to_G-G_P_max)/G_P_max_b_parameter)  - G_P_max_b_parameter*F.softplus((-G_P_max)/G_P_max_b_parameter)
            G_P_excess = torch.clamp(G_P_excess, min=torch.tensor(0.0, device=water_going_to_G.device), max=water_going_to_G)
     
            G_exces.append(G_P_excess)
            water_after_excess_G=water_going_to_G-G_P_excess
            ##===Calcul Overflow from Canopy Bucket
            
            f_wet_G.append(torch.min(torch.tensor(1.0), G_level[i]/(Gmax_dynamic[i]+0.0001)))
            ##==Calculating ET loss from canopy bucket
            G_input_NN=torch.tensor([(G_level[i] / G_max), T_air_scaled[i], Deficit_scaled[i], Rn_scaled[i],delt_scaled[i],um_scaled[i] ]).unsqueeze(0) 
            G_NN_output=self.G_ET_surrogate(G_input_NN).squeeze()
            
            
            #=======PM Equation Component=== Start ====
            ra_CC=torch.sigmoid (-C_ra_a_param*torch.log1p(um_scaled[i])+self.C_ra_b_theta)*100 + 5
            ra_C.append(ra_CC)
            E_interc_energy=(torch.relu(Rn[i])*delt[i])/(delt[i]+gama[i])  + ((1.23 * 1013 * Deficit[i]) / ra_C[i]) / (delt[i] + gama[i])
            E_interc = E_interc_energy*(86400/lamda[i])

            E_G_loss_potential=E_interc-torch.relu(E_interc-G_level[i])#torch.sigmoid(F.softplus(G_b1)*torch.log1p(G_level[i]/G_max)+F.softplus(G_b2)*torch.log1p(E_interc/G_max)-self.G_a1_theta)

            #=======PM Equation Component=== End =======
            
            #E_G_loss_potential=torch.sigmoid(F.softplus(G_b1)*torch.log1p(G_level[i]/G_max)+F.softplus(G_b2)*torch.log1p(G_NN_output/G_max)-self.G_a1_theta)
            #E_G_loss_potential=torch.sigmoid(F.softplus(G_b1)*torch.log1p(G_level[i]/G_max)+F.softplus(G_b2)*torch.log1p(Rn_scaled[i])+F.softplus(G_b3)*torch.log1p(T_air_scaled[i])+F.softplus(G_b4)*torch.log1p(delt_scaled[i])+F.softplus(G_b5)*torch.log1p(Deficit_scaled[i])+F.softplus(G_b6)*torch.log1p(um_scaled[i])-self.G_a1_theta)
           
            G_loss=E_G_loss_potential#*(f_veg[i])*(f_wet_G[i])
            L_G.append(G_loss)

            water_balance_in_G=G_level[i]+water_melted_G1[i]+water_after_excess_G-L_G[i]
            G_Overflow =G_max_b_parameter*F.softplus((water_balance_in_G-Gmax_dynamic[i])/G_max_b_parameter)  - G_max_b_parameter*F.softplus((-Gmax_dynamic[i])/G_max_b_parameter)
            G_Overflow = torch.clamp(G_Overflow, min=torch.tensor(0.0, device=G_Overflow.device), max=water_balance_in_G)
     

            OF_G.append(G_Overflow)
            water_balance_in_G_after_OF=water_balance_in_G-OF_G[i]
            ###=====Saving Bucket level for next time step
            #print(G_max_b_parameter,water_balance_in_G,Gmax_dynamic[i],G_Overflow )
            if G_level[i]<0:   
                
                break
            
            G_level.append(water_balance_in_G_after_OF)        
            #print (water_after_excess_G, water_balance_in_G, E_G_loss_potential, G_level[i],water_balance_in_G_after_OF, OF_G[i],  )

            
        
            water_coming_to_soil_level=Precip[i]*(1-f_veg[i])+OF_G[i]+G_P_excess
            
            ####++=======Soil Bucket++==============###
            water_going_to_S1=Precip[i]*(1-f_veg[i])*f_snow[i]+OF_G1[i]
            water_melted_S1.append(S1_level[i]*f_melt_soil[i])
            water_balance_in_S1=S1_level[i]+water_going_to_S1-water_melted_S1[i]
            S1_level.append(water_balance_in_S1)
            
            ##====Calculating Infilitration Excess from Soil bucket
            water_moving_to_S=Precip[i]*(1-f_veg[i])*f_rain[i]+OF_G[i]+water_melted_S1[i]
            S_P_excess=S_P_max_b_parameter*F.softplus((water_moving_to_S-S_P_max)/S_P_max_b_parameter)  - S_P_max_b_parameter*F.softplus((-S_P_max)/S_P_max_b_parameter)
            S_P_excess = torch.clamp(S_P_excess, min=torch.tensor(0.0, device=water_moving_to_S.device), max=water_moving_to_S)
     
            S_exces.append(S_P_excess)
            water_after_excess_S=water_moving_to_S-S_P_excess
            ##===Calcul Overflow from Soil Bucket
            
            S_input_NN=torch.tensor([S_level[i]/S_max, T_air_scaled[i], Deficit_scaled[i], Rn_scaled[i],delt_scaled[i],um_scaled[i] ]).unsqueeze(0) 
            S_NN_output=self.S_ET_surrogate(S_input_NN).squeeze()
            
            
            #=======PM Equation Component=== Start ====
            ra_SS=torch.sigmoid (-S_ra_a_param*torch.log1p(um_scaled[i])+self.S_ra_b_theta)*100 + 5
            ra_S.append(ra_SS)
            rs_SS=S_NN_output#torch.sigmoid(-torch.exp(self.S_rs_X)*(S_level[i]/S_max)-torch.exp(self.S_rs_Rn)*Rn_scaled[i]-torch.exp(self.S_rs_VPD)*Deficit_scaled[i]-torch.exp(self.S_rs_T)*T_air_scaled[i]+ self.S_rs_b)*1000 + 5
            rs_S.append(rs_SS)
            E_soil_energy = (
                    (torch.relu(Rn[i]) * delt[i]) + 
                    ((1.23 * 1013 * Deficit[i]) / ra_S[i])
                ) / (delt[i] + gama[i] * (1 + (ra_S[i] / (1+rs_S[i]))))            
            
            E_soil = E_soil_energy* (86400/lamda[i])

            E_S_loss_potential=E_soil-torch.relu(E_soil-S_level[i])#torch.sigmoid(F.softplus(S_b1)*torch.log1p(S_level[i]/S_max)+F.softplus(S_b2)*torch.log1p(E_soil/S_max)-self.S_a1_theta)

            #=======PM Equation Component=== End ====
            
            #E_S_loss_potential=torch.sigmoid(F.softplus(S_b1)*torch.log1p(S_level[i]/S_max)+F.softplus(S_b2)*torch.log1p(S_NN_output/S_max)-self.S_a1_theta)
            #E_S_loss_potential=torch.sigmoid(F.softplus(S_b1)*torch.log1p(S_level[i]/S_max)+F.softplus(S_b2)*torch.log1p(Rn_scaled[i])+F.softplus(S_b3)*torch.log1p(T_air_scaled[i])+F.softplus(S_b4)*torch.log1p(delt_scaled[i])+F.softplus(S_b5)*torch.log1p(Deficit_scaled[i])+F.softplus(S_b6)*torch.log1p(um_scaled[i])-self.S_a1_theta)
            
            S_loss=(E_S_loss_potential *omega_S_ET)#*(1-f_veg[i])
            L_S.append(S_loss)
            
            ##====Calculating Gates=======
            Gate_S_out=omega_S_O*torch.sigmoid(F.softplus(self.S_b_out_X)*torch.log1p(S_level[i]/S_max)-self.S_a_out)
            Gate_S_seepage=omega_S_S*torch.sigmoid(F.softplus(self.S_b_seepage_X)*torch.log1p(S_level[i]/S_max)-self.S_a_seepage)
            
            ###=====Calculating Fluxes======
            Seepage_S.append(Gate_S_seepage*S_level[i])
            O_S.append(Gate_S_out*S_level[i])
            
            water_balance_in_S=S_level[i]+water_after_excess_S-(O_S[i]+L_S[i]+Seepage_S[i])
            
            S_Overflow=S_max_b_parameter*F.softplus((water_balance_in_S-S_max)/S_max_b_parameter)-S_max_b_parameter*F.softplus((-S_max)/S_max_b_parameter)
            S_Overflow = torch.clamp(S_Overflow, min=torch.tensor(0.0, device=G_Overflow.device), max=water_balance_in_S)
            OF_S.append(S_Overflow); 
            
            water_balance_in_S_after_OF=water_balance_in_S-OF_S[i]

            ###====Updating State for next time step=====
            S_level.append(water_balance_in_S_after_OF)
            
            
            ####++=======Baseflow Bucket++==============###

            ##====Loss from B bucket=====###3
            B_input_NN=torch.tensor([B_level[i]/200, T_air_scaled[i], Deficit_scaled[i], Rn_scaled[i],delt_scaled[i],um_scaled[i] ]).unsqueeze(0) 
            B_NN_output=self.B_ET_surrogate(B_input_NN).squeeze()
            
            #=======PM Equation Component=== Start ====
            rs_CC=B_NN_output#torch.sigmoid(-torch.exp(self.C_rs_X)*(B_level[i]/S_max)-torch.exp(self.C_rs_Rn)*Rn_scaled[i]-torch.exp(self.C_rs_VPD)*Deficit_scaled[i]-torch.exp(self.C_rs_T)*T_air_scaled[i]+ self.C_rs_b)*1000 + 5
            rs_C.append(rs_CC)
            E_transp_energy = (
                    ( torch.relu( Rn[i]) * delt[i]) + 
                    ((1.23 * 1013 * Deficit[i]) / ra_C[i])
                ) / (delt[i] + gama[i] * (1 + (ra_C[i] / (1+rs_C[i])))) 
            
            E_transp = E_transp_energy*  (86400/lamda[i])
            E_B_loss_potential=E_transp-torch.relu(E_transp-B_level[i])#torch.sigmoid(F.softplus(B_b1)*torch.log1p(B_level[i]/200)+F.softplus(B_b2)*torch.log1p(E_transp/S_max)-self.B_a1_theta)
            #=======PM Equation Component=== End ====
            
            
            #E_B_loss_potential=torch.sigmoid(F.softplus(B_b1)*torch.log1p(B_level[i]/200)+F.softplus(B_b2)*torch.log1p(B_NN_output/200)-self.B_a1_theta)
            #(Rn_scaled[i])+F.softplus(B_b3)*torch.log1p(T_air_scaled[i])+F.softplus(B_b4)*torch.log1p(delt_scaled[i])+F.softplus(B_b5)*torch.log1p(Deficit_scaled[i])+F.softplus(B_b6)*torch.log1p(um_scaled[i])-self.B_a1_theta)
            
            B_loss=(E_B_loss_potential   * omega_B_ET)#*f_veg[i]*(1-f_wet_G[i])
            L_B.append(B_loss)
            
            ##=====Outflow from B Bucket=====###
            Gate_B_out=omega_B_O*torch.sigmoid(F.softplus(self.B_b_out_X)*torch.log1p(B_level[i]/200)-self.B_a_out)
            O_B.append(Gate_B_out*B_level[i])
            Remembr_B = B_level[i]+Seepage_S[i]-(O_B[i]+L_B[i])
            B_level.append(Remembr_B)
            
            ####++=======Routing Delay Bucket++==============###
            #Gate_D_state_rev=D_level[i]+O_S[i]+OF_S[i]+P_exces[i]
            #Gate_D_out=torch.sigmoid(F.softplus(self.D_b_out_X)*torch.log1p(Gate_D_state_rev/S_max)- self.D_a_out)
            #O_D.append(Gate_D_out*Gate_D_state_rev)
            #Gate_D_remember=1-Gate_D_out
            #D_level.append(Gate_D_remember*Gate_D_state_rev)#+P_exces[i]+OF_S[i]
            
            #XG1.append(G1_level[i]); XS1.append(S1_level[i]); P.append(Precip[i]);  GSE.append(S_P_excess/(water_moving_to_S+eps)); ES.append(S_P_excess); IS.append(water_moving_to_S-S_P_excess); LS.append(L_S[i]); GSL.append(S_loss); XS.append(S_level[i]); GSO1.append(S_Overflow/(water_balance_in_S+eps)); O1S.append(S_Overflow); GSO2.append(Gate_S_out); O2S.append(O_S[i]); GSO3.append(Gate_S_seepage); O3S.append(Seepage_S[i]); XB.append(B_level[i]); GBL.append(B_loss); LB.append(L_B[i]); GBO.append(Gate_B_out); OB.append(O_B[i]); O.append(O_S[i]+O_B[i]+S_exces[i]+OF_S[i])     #XD.append(D_level[i]); GDO.append(Gate_D_out) ; OD.append(O_D[i])         
            #WB_G.append(water_balance_in_G)
        '''
        model_params = {
                "b_rain_fraction_theta": F.softplus(self.b_rain_fraction_theta),
                "c_rain_fraction_theta": self.c_rain_fraction_theta,
                "b1_melting_canopy_theta": F.softplus(self.b1_melting_canopy_theta),
                "b2_melting_canopy_theta": F.softplus(self.b2_melting_canopy_theta),
                "c_melting_canopy_theta": self.c_melting_canopy_theta,
                "b1_melting_soil_theta": F.softplus(self.b1_melting_soil_theta),
                "b2_melting_soil_theta": F.softplus(self.b2_melting_soil_theta),
                "c_melting_soil_theta": self.c_melting_soil_theta,
                "G_max": G_max ,
                "G_max_b_parameter": G_max_b_parameter,
                "G_P_max": G_P_max,
                "G_P_max_b_parameter": G_P_max_b_parameter,
               
                "S_max": S_max,
                "S_max_b_parameter": S_max_b_parameter,
               
                "S_P_max": S_P_max,
                "S_P_max_b_parameter": S_P_max_b_parameter,

                "omega_S": omega_S,
                "omega_S_O": omega_S_O,
                "omega_S_S": omega_S_S,
                "omega_S_R": omega_S_R,
                "omega_S_ET": omega_S_ET,
                "omega_B": omega_B,
                "omega_B_O": omega_B_O,
                "omega_B_R": omega_B_R,
                "omega_B_ET": omega_B_ET,
                "G_b1": F.softplus(G_b1),
                "G_b2": F.softplus(G_b2),
                "G_a1_theta": self.G_a1_theta,
                "S_b1": F.softplus(S_b1),
                "S_b2": F.softplus(S_b2),
                "S_a1_theta": self.S_a1_theta,
                "S_b_out_X": F.softplus(self.S_b_out_X),
                "S_a_out": self.S_a_out,
                "S_b_seepage_X": F.softplus(self.S_b_seepage_X),
                "S_a_seepage": self.S_a_seepage,
                "B_b1": F.softplus(B_b1),
                "B_b2": F.softplus(B_b2),
                "B_a1_theta": self.B_a1_theta,
                "B_b_out_X": F.softplus(self.B_b_out_X),
                "B_a_out": self.B_a_out
                
        }

        diagnostics = {
            "P": torch.stack(P),            # Precipitation
            "f_snow": torch.stack(f_snow),            # Precipitation
            "GSE": torch.stack(GSE),        # Gate: soil excess
            "ES": torch.stack(ES),          # Excess from soil
            "IS": torch.stack(IS),          # Infiltration into soil
            "LS": torch.stack(LS),          # Soil loss
            "GSL": torch.stack(GSL),        # Gate: soil loss
            "XS": torch.stack(XS[:-1]),          # Soil level
            
            "XG1": torch.stack(XG1[:-1]),
            "XS1": torch.stack(XS1[:-1]),
            
            "G_max_dynamic":torch.stack(Gmax_dynamic),
            "WB_G":torch.stack(WB_G),
            "XG":torch.stack(G_level[:-1]),
            "LG":torch.stack(L_G),
            "GE":torch.stack(G_exces),
            "G_OF":torch.stack(OF_G),
            "PG":torch.stack(P_G),
            
            "GSO1": torch.stack(GSO1),      # Gate: soil overflow (part 1)
            "O1S": torch.stack(O1S),        # Overflow part 1
            "GSO2": torch.stack(GSO2),      # Gate: soil outflow (part 2)
            "O2S": torch.stack(O2S),        # Outflow part 2
            "GSO3": torch.stack(GSO3),      # Gate: soil seepage (part 3)
            "O3S": torch.stack(O3S),        # Seepage from soil
        

        
            "XB": torch.stack(XB[:-1]),          # Baseflow layer level
            "GBL": torch.stack(GBL),        # Gate: baseflow loss
            "LB": torch.stack(LB),          # Loss from baseflow
            "GBO": torch.stack(GBO),        # Gate: baseflow outflow
            "OB": torch.stack(OB),          # Outflow from baseflow
        
            "O": torch.stack(O)             # Total outflow (OD + OB)
        }
        '''
        #print(max(C_level[i]))
        S_exces_tensor=torch.stack(S_exces)

        OF_S_tensor=torch.stack(OF_S)

        #O_D_tensor=torch.stack(O_D)
        O_B_tensor=torch.stack(O_B)
        O_S_tensor=torch.stack(O_S)
        
        return S_exces_tensor+OF_S_tensor+O_B_tensor+O_S_tensor#, diagnostics, model_params




