%% Optimization Code with Cogeneration and PV constraint
clear all; clc; close all

% helps you set options for your optimization problem.
options=optimoptions('intlinprog','MaxTime',300) ;       % this is used to limit the solution time to max. 180 sec

n=288;                                                  % time interval 

day=216;                                                % day of the year

z=zeros(n,n);
t=tril(ones(n,n),0);
e=eye(n,n);
delta_t=1/12;                                          % time interval of 5 minutes converted to hours (5 / 60)

aeq = ones(1, n); %vector for equality constraint
z_ = zeros(1, n); %zeros vector



%% ********* Electricity prices


%% *** TOU PRICING SCHEME




T1=0.187;                                              % TOU rate 06:00-17:00 (11hrs = )
T2=0.278;                                              % TOU rate 17:00-22:00 (5 hrs)
T3=0.114;                                               % TOU rate 22:00-06:00 (8 hrs) % Represents 5 miniutes intervals in hours (5 / 60)

%{

price=[T3*ones(1,12) T3*ones(1,12) T3*ones(1,12)...
    T3*ones(1,12) T3*ones(1,12) T3*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T1*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T1*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T1*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T2*ones(1,12)...         % Each hour represents 12 price time slots of 5 minutes interval.
    T2*ones(1,12) T2*ones(1,12) T2*ones(1,12)...        %price is vector has dimension [288 1] which represents price for 24 hrs expressed every 5 minutes interval (24 *60 / 5 = 288)
    T2*ones(1,12) T3*ones(1,12) T3*ones(1,12)]';        % electricity buying price


pr_buy=price*delta_t;

%}


%{
%% ****REAL-TIME PRICING SCHEME****



price=1.03*[0.11*ones(1,12) 0.08*ones(1,12) 0.065*ones(1,12)...
    0.05*ones(1,12) 0.065*ones(1,12) 0.11*ones(1,12)...
    0.23*ones(1,12) 0.20*ones(1,12) 0.16*ones(1,12)...
    0.14*ones(1,12) 0.16*ones(1,12) 0.21*ones(1,12)...
    0.18*ones(1,12) 0.17*ones(1,12) 0.16*ones(1,12)...
    0.18*ones(1,12) 0.20*ones(1,12) 0.24*ones(1,12)...         % Each hour represents 12 price time slots of 5 minutes interval.
    0.31*ones(1,12) 0.37*ones(1,12) 0.32*ones(1,12)...        %price is vector has dimension [288 1] which represents price for 24 hrs expressed every 5 minutes interval (24 *60 / 5 = 288)
    0.24*ones(1,12) 0.16*ones(1,12) 0.12*ones(1,12)]';        % electricity buying price



pr_buy = price*delta_t;

%}


%% ***** CPP PRICING SCHEME *******

%% 5-hour CPP Price signal

Tcpp3 = T3 - 0.1*T3;
Tcpp1 = T1 - 0.1*T1;
%Tcpp2 = T2 - 0.1*T2;
Tcpp = 3*T2;

price=[Tcpp3*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)...
    Tcpp3*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp*ones(1,12)...         % Each hour represents 12 price time slots of 5 minutes interval.
    Tcpp*ones(1,12) Tcpp*ones(1,12) Tcpp*ones(1,12)...        %price is vector has dimension [288 1] which represents price for 24 hrs expressed every 5 minutes interval (24 *60 / 5 = 288)
    Tcpp*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)]';        % electricity buying price




pr_buy = price*delta_t;


%% 2-hour CPP Price signal
%{
Tcpp3 = T3 - 0.1*T3;
Tcpp1 = T1 - 0.1*T1;
Tcpp2 = T2 - 0.1*T2;
Tcpp = 3*T2;


price=[Tcpp3*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)...
    Tcpp3*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp*ones(1,12)...
    Tcpp*ones(1,12) Tcpp1*ones(1,12) Tcpp2*ones(1,12)...         % Each hour represents 12 price time slots of 5 minutes interval.
    Tcpp2*ones(1,12) Tcpp2*ones(1,12) Tcpp2*ones(1,12)...        %price is vector has dimension [288 1] which represents price for 24 hrs expressed every 5 minutes interval (24 *60 / 5 = 288)
    Tcpp2*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)]';        % electricity buying price



pr_buy=price*delta_t;

%}


%% FLAT-RATE PRICING SCHEME
%flat_price = 0.185*ones(1,288)';
%flat_pr_buy =  flat_price*delta_t;
%price = 0.185*ones(1,288)';
%pr_buy =  price*delta_t;


%% Feed-in Tariff rate
FiT=0.169462;
pr_sell=FiT*ones(n,1)*delta_t;                          % electricity selling price 

%% ******End of Pricing Scheme

%% *********Istanbul - Turkey Temperature Related Data*******
% Outdoor temperature
% �stanbul, Turkey

ist=xlsread('isttemp2')';                               % annual air temperature in 2018 (oC)
To_ann=zeros(1,52560*2);
for i=1:52560
 To_ann(i*2:i*2+2)=ist(i);   
end
To_ann=To_ann(2:52560*2);
To_ann=[To_ann To_ann(end)];
Tout=To_ann(day*n+1:(day+1)*n);                         % daily temperature on n'th day            

%sonrasil
Tout=Tout+0.2;

%% ********PV Solar production *******
%PV model

% PV panel electrical data
Y_pv=230;                                               % the rated capacity of the PV array (power output under STC) [W]
f_pv=95;                                                % PV derating factor [%]
P_coeff=-0.0038;                                        % temperature coefficient of power [%/oC].
% PV power output
tilt=29;                                                % PV tilt angle 
lat=41;                                                 % latitude of the region
roug=0.2;                                               % ground albedo (ground reflectance)


%Solar data
sol_data=xlsread('istrad2')';
Rad_ann=zeros(1,52560*2);



for i=1:52560
    Rad_ann(i*2:i*2+2)=sol_data(i);
end

solar_data=Rad_ann(2:52560*2); 


GHI=solar_data(1,:);                                    % global horizontal irradiance  

Isc=1353;
d2r=pi/180;                                             % degree to radian conversion
H=GHI;
Gsc=1353;                                               % the solar constant [1367 W/m2]

for nd=1:364                                            % number of days in a year
    
    dec(1,nd)=23.45*sind(360*(284+nd)/365);             % solar declination [degree] 
    
    ws(1,nd) = acosd(tand(dec(1,nd))*(-tand(lat)));     % the hour angle [degree]
    
 if ws(1,nd) < acosd(tand(dec(1,nd))*(-tand(lat-tilt)));
     wsp(1,nd)=ws(1,nd);                                % the hour angle for tilted surfaces of the day under consideration [degree]
 else
     wsp(1,nd)= acosd(tand(dec(1,nd))*(-tand(lat-tilt)));
 end
    
 for h=1:24*(1/delta_t)
       m=24*(1/delta_t)*(nd-1)+h;

Rb(1,m)= ( cosd(lat-tilt)*cosd(dec(1,nd))*sind(wsp(1,nd)) + wsp*pi/180*sind(lat-tilt)*sind(dec(1,nd)) )...
        /( cosd(lat)*cosd(dec(1,nd))*sind(ws(1,nd)) + ws*pi/180*sind(lat)*sind(dec(1,nd)) );
    
    Gon(1,nd)=Gsc*(1+0.033*cosd(360*nd/365)); 
    Ho(1,m)=24*Gon(1,nd)/pi*(cosd(lat)*cosd(dec(1,nd))...
        *sind(ws(1,nd))+2*pi*ws(1,nd)/360*sind(lat)...
        *sind(dec(1,nd)));                              % extraterrestrial solar radiation received over a day on a horizontal surface [!!!]
        
    Kt(1,m)=H(1,m)/Ho(1,m);                             % clearness index
    
    Hd(1,m)=H(1,m)*(1-1.13*Kt(1,m));                    % the total diffuse radiation on tilted surface [W/m2]
    
    R(1,m) = (1-Hd(1,m)/H(1,m))*Rb(1,m)  +  Hd(1,m)*(1+cosd(tilt))/(2*H(1,m)) + roug*(1-cosd(tilt))/2;

    Ht(1,m) = R(1,m)*H(1,m);                            % the total incident radiation on tilted surface [W/m2]

end
end

Ht(isnan(Ht))=0;                                        % converts NaN values into zeroes in "Ht"
Gt=Ht;
Gt= Gt(day*24*(1/delta_t)+1:(day+1)*24*(1/delta_t));    % selection of daily data from yearly data
Gt_stc = 1000;                                          % the incident radiation at STC [W/m2]

% The PV cell temperature in the current time step [oC]
TcNOCT=45;                                              % the nominal operating cell temperature [oC]
TcSCT=25;                                               % the PV cell temperature under STC [oC]
TaNOCT=20;                                              % the ambient temperature at which the NOCT is defined [oC]
GtNOCT=800;                                             % the solar radiation at which the NOCT is defined [W/m2]             
                              
Tc=Tout+(TcNOCT-TaNOCT)*Gt/GtNOCT;                      % PV cell temperature [oC]

% Power output of the PV array.
P_pv=Y_pv*f_pv/100*(Gt/Gt_stc).*[1+P_coeff*(Tc-TcSCT)]; % PV production of a module (230 W) [kW]

n_pv= 172;                                                % number of PV modules.
PVprod=-P_pv*n_pv/1000;                                 % distribution of the PV production (kW) 
PVprod(isnan(PVprod))=0;   

%PVprod = zeros(1,288);
solar = PVprod;

%% End of PV Production



%% ********** BESS SYSTEM ************       

%BES parameteres
n_bess = 10; %number of UPS batteries

%R_bes=0.005;            %%used when there is no battery  % BES charging rate (kW)
R_bes=3.3*(n_bess);                                              % BES charging rate (kW)

D_bes = 3.3*(n_bess);                                              %BES discharging rate (kW)
CE_bes= 0.95;                                            % charging efficiency of BES
DE_bes= 0.95;                                            % discharging efficiency of BES

eff_rt_bes=sqrt(CE_bes*DE_bes);                         % round-trip efficiency of BES battery 89%
                                                        % fraction of energy put into the storage that can be retrieved. Typically it is about 80%. 
Cyc_bes=2000;                                           % BES battery lifetiisce in cycles - Each round of full discharge and then recharge.
       
DoD_bes=0.8;                                            % depth of discharge for which lifetime in cycles is determined                      


%Cap_bes=0.00006;        %used when there is no battery  % BES battery capacity (kWh) 
Cap_bes= 24*n_bess ;                                           % BES battery capacity (kWh)

Life_bes=Cyc_bes*Cap_bes*DoD_bes;                       % battery lifetime throughput energy measured for specific DoD
%DoD indicates the percentage of the battery that has been discharged relative to the overall capacity of the battery

Life_bes=Life_bes;                           
c_rep_bes=2850.0;                                        % battery replacement cost including labor ($)

deg_cost_bes=c_rep_bes/(Life_bes*eff_rt_bes);            % degradation cost of ESS battery ($/kWh)
 % (Life_bes*eff_rt_bess equals the total amount of retrievable energy
 % during the entire life time of the BESS. Degradaton cost simply
 % highlights the price you pay for each discharge round of the battery

deg_cost_bes=deg_cost_bes*delta_t;

%buy price considering degradation cost                                                 
pr_buy_ESS = pr_buy+deg_cost_bes';                        % electricity buying price for BES considering battery degradation ($/kWh)                                                         
pr_sell_ESS=pr_sell-deg_cost_bes';               % electricity sell back price of ESS($/kWh)

pr_buy_ESS_pv = ones(n,1)*deg_cost_bes;

bes_max=Cap_bes/delta_t;                                % maximum SOE of BES (kWh)
bes_min=Cap_bes*(1-DoD_bes)/delta_t;                    % minimum SOE of BES(kWh)
bes_ini= bes_min;                                        % initial SOE of BES (kWh) 

DMax_bes=bes_max-bes_ini;                              % daily requirement of BES (kWh)
DMin_bes=bes_ini-bes_min;                               % daily min. allowed capacity of BES (kWh)                               % daily min. allowed capacity of BES (kWh)

%% END OF BESS SYSTEM


%% ************ WORKLOAD *************

%import workloads from python generated script
workloads = load('2flex_workloads.mat');

%Load Time-shiftable workload #1

p1 = workloads.flex1;

PP1(1,:)= p1(:);                     
for k=1:n-1
PP1(k+1,:)=circshift(p1,k);                               % shifted profile possibilities of TSA1
end 

% Load Time-shiftable Workload #2 (a1-2)
p2 = workloads.flex2;   % energy consumption profile of TSA2


PP2(1,:)= p2(:); 
for k=1:n-1
PP2(k+1,:)=circshift(p2,k);                               % shifted profile possibilities of TSA2
end 

start_time = 0;

P1 = limit_execution(p1 ,PP1, start_time);

P2 = limit_execution(p2 ,PP2, start_time);

% Load inflexible workload profile 
low = 10; %kW
high = 50; %kW
numElements = 288; %timestep

max_flex = max(p1 + p2);
min_flex =  min(p1 + p2);



inflex  =  [50*ones(1,12) 50*ones(1,12) 50*ones(1,12)...
           50*ones(1,12) 50*ones(1,12) 50*ones(1,12)...
           100*ones(1,12) 100*ones(1,12) 100*ones(1,12)...
           25*ones(1,12) 25*ones(1,12) 25*ones(1,12)...
           25*ones(1,12) 25*ones(1,12) 25*ones(1,12)...
           50*ones(1,12) 50*ones(1,12) 100*ones(1,12)...
           50*ones(1,12) 50*ones(1,12) 50*ones(1,12)...
           25*ones(1,12) 25*ones(1,12) 25*ones(1,12)]; %CRAH outlet temperature for each hour

%% End of workload




%% ************* CRAH UNIT *****************
n_racks = 80;
n_servers = 80*42;  % #number of servers per rack
mass_rate = n_servers * 0.02274; % #42 * 0.02274 #mass flow rate for 42 servers
ht_cap = 1.005;% #KJ/Kg.k #specific heat capacity

C_dot  =  mass_rate * ht_cap;

cr = C_dot; 

leak = 0.25;

cc = cr / (1 - leak);
al = 0.176;
%cc = 0.5; %kg/s %kW/K
clk = cc*leak;
ca = cc- clk; 
cal = ca*al ; 
crec = cal;
Wmisc = 20; %kW 10% of total DC load (1MW)

%pcrac = (tci - Tc_o)*cc / COP; 

COP = 3.9; %

fcop = cc/COP;

%cr =ca ;


c1 = crec + cc ;
c2 = clk + cal;
c3 = cr - crec;

% constants
tau = 396; %seconds
deltt = (1/12)*3600; % convert 5 minutes time interval to seconds = 300s
eps = 0.89; %epsilon

c6 = tau/(tau + deltt);
c5 = 1/cr;
c4  = deltt / (tau+deltt);
c7 = c5*c6;

c8 = (c6*eps - 1);
c9 = c7-c5;

Tco1=14.5;                                              % Tc_o at 06:00-17:00 (11hrs = )
Tco2=15.0;                                              % Tc_o at 17:00-22:00 (5 hrs)
Tco3=14.0;                                             % Tc_o at 22:00-06:00 (8 hrs) 
Tco4=16.0;                                                       % Represents 5 miniutes intervals in hours (5 / 60)

Tro_max = 25 ; % set the maximum rack outlet temperature


Tc_o  =  [Tco2*ones(1,12) Tco4*ones(1,12) Tco1*ones(1,12)...
           Tco1*ones(1,12) Tco1*ones(1,12) Tco2*ones(1,12)...
           Tco1*ones(1,12) Tco1*ones(1,12) Tco2*ones(1,12)...
           Tco1*ones(1,12) Tco1*ones(1,12) Tco2*ones(1,12)...
           Tco1*ones(1,12) Tco1*ones(1,12) Tco2*ones(1,12)...
           Tco1*ones(1,12) Tco1*ones(1,12) Tco2*ones(1,12)...
           Tco1*ones(1,12) Tco1*ones(1,12) Tco2*ones(1,12)...
           Tco1*ones(1,12) Tco1*ones(1,12) Tco2*ones(1,12)]'; %CRAH outlet temperature for each hour


%Tc_o = 18*ones(n , 1);
%Decision variables
Tr_o = 0.0;
Tr_i = 0.0;
Tc_i = 0.0; 
Ts = 0.0;

Ts0 = 16;

%% constants
tau = 396; %seconds
deltt = (1/12)*3600; % convert 5 minutes time interval to seconds = 300s
eps = 0.89; %epsilon

k = tau/(tau + deltt);
theta = deltt / (tau+deltt);


tt = [e(: ,end)' ; e(1:end-1 , : )] - e*k ; % for Ts constraint (5)


%% Generate Ts Time Dependent constraint Block -Equation 4

rw = [[-c6, 1] , zeros(1,n-2)];

%first row of constraint
TSS(1,:) = [[1], zeros(1, n-1)];

for i=1:n-1
    %the remaining time dependent constraints from t=2 to t = n for the Ts
    %, Ts-1 related constraint
    TSS(i+1, :) = circshift(rw,i-1) ;      %
end 


%b rhs of the above constraint
bt = [c6*Ts0 , zeros(1,n-1)];


%% Generate  Time dependent constraints for Tro
er = [[0], zeros(1, n-1)];

ts = -c6*eps*eye(n);

%ts(1,:) = zeros(1,n);
TSS2 = [er; ts(1:end-1,:)] ;  % Handle Ts-1 
%TSS2 = ts;


%b rhs of the above constraint
bb = [c6*eps *Ts0 , zeros(1,n-1)]; %for the TS - c4TS-1 time dependent constraint

%Limit the TRO < Tmax constraint 


%% COGENERATION SYSTEM SUPPLY POWER - ELECTRICITY

cogen_price =  0.189*ones(1,n)'; 
cogen_pr_buy =  cogen_price*delta_t;
cogen_pr_buy_ESS = cogen_pr_buy +deg_cost_bes';



%cogen heat selling price
heat_pr=  0.075*ones(1,n)';
heat_sell =  heat_pr*delta_t;

%Co-generation Electricity Selling Price
cogen_sell  =  FiT*ones(1,n)' ; %0.133*ones(1,288)'; %Turkish lira equivalent of US$0.133 per kWh for biomass power plants; and

cogen_pr_sell = cogen_sell*delta_t;

max_cogen = 500;%kW % Maximum available cogen power

%% CO-GENERATION HEATING DEMAND PROFILE
h1=50;                                            
h2=100;     %kW                                        
h3=75;                                            

% Heating profile
cogen_heat = [h2*ones(1,12) h2*ones(1,12) h2*ones(1,12)...
               h2*ones(1,12) h2*ones(1,12) h2*ones(1,12)...
               h3*ones(1,12) h3*ones(1,12) h3*ones(1,12)...
               h1*ones(1,12) h1*ones(1,12) h1*ones(1,12)...
               h1*ones(1,12) h1*ones(1,12) h1*ones(1,12)...
               h3*ones(1,12) h3*ones(1,12) h3*ones(1,12)...
               h2*ones(1,12) h2*ones(1,12) h2*ones(1,12)...
               h2*ones(1,12) h2*ones(1,12) h2*ones(1,12)]';

cgof = 1.0; %cogenerator switch

%% *** ABSORPTION CHILLER ****
eta_mt = 0.3;
rf_ac = 0.7;
rf_dh = 0.9;

COP_ACH = 1.3 ; % Thermal COP of Absorption chiller
COP_ACE = 20; % Electricity COP of Absorption chiller

%Q_ac_max = P_cogen *(1 - eta_mt)/ (eta_mt*rf_ac*COP_ACH) ; 
Q_ac_coef = (1 - eta_mt)/ (eta_mt*rf_ac*COP_ACH) ; 

Q_ach_coef = (1 - eta_mt)/ (eta_mt*rf_dh) ; 


%% *** ECONOMIZER
COP_ECO = 20 ; % Economizer COP

t1 = 10; %kW - cool summer 
t2 = 20; %kW - mild winter
t3 = 15; %kW

Q_eco_max = [t2*ones(1,12) t2*ones(1,12) t2*ones(1,12)...
               t2*ones(1,12) t2*ones(1,12) t2*ones(1,12)...
               t3*ones(1,12) t3*ones(1,12) t3*ones(1,12)...
               t1*ones(1,12) t1*ones(1,12) t1*ones(1,12)...
               t1*ones(1,12) t1*ones(1,12) t1*ones(1,12)...
               t3*ones(1,12) t3*ones(1,12) t3*ones(1,12)...
               t2*ones(1,12) t2*ones(1,12) t2*ones(1,12)...
               t2*ones(1,12) t2*ones(1,12) t2*ones(1,12)]';

%% ** THERMAL ENERGY STORAGE SYSTEM

COP_TES = 1000; 

%% *** CRAC UNIT 
max_crac  = 200 ; %kW (CRAC unit rated power)

%miscillaneous load
misc = Wmisc*ones(1,n)'; 



M = 10000000;


%------------------------------------------------------------------------------------------------------------------------------------------------(------Pcrac_tot-------)
%%----Grid2BESS-----Cogen2BESS------BESS_Sell----PVSell---Grid2Inflex-------Cogen2Inflex--------Flex1------------Flex2-----------Grid2Flex-------Cogen2Flex------------Tro----------------Tri-----------------Tci-------------Tco--------------Ts-------------Grid2CRAC---------Cogen2CRAC-------misc------cogen2misc------PV2BESS-----------BESS_ON_OFF---PV2Inflex----PV2Flex-----PV2CRAC---------PV2Misc---------HeatSell---------C2Grid
f = [pr_buy_ESS;   cogen_pr_buy;  pr_sell_ESS; pr_sell;     pr_buy;         cogen_pr_buy;    P1*pr_buy*0.0; P2*pr_buy*0.0;  pr_buy*1.001;     cogen_pr_buy*1.001;  pr_buy*Tr_o*0.00; pr_buy*Tr_i*0.00;  pr_buy*0.0 ;        pr_buy *0.0  ;  pr_buy*Ts*0.00 ;   pr_buy*1.0 ;   cogen_pr_buy*1.0 ;  pr_buy*1.0 ; cogen_pr_buy ; pr_buy_ESS_pv  ; pr_buy*0.0 ; pr_buy*0.0 ; pr_buy*0.0 ; pr_buy*0.0 ; pr_buy*0.0 ;  heat_sell*1.0    ;    cogen_pr_sell*1];   % decision variable cost coefficients        



intcon = [n*6 + 1 : n*6+n+n  n*20+1 : n*20+n];    % the orders of decision variables that are integefigurer   

pv_bess_ub = min(R_bes*ones(1,n) , -solar); %u

%--------1Grid2BESS--------2Cogen2BESS---------3BESS2G---------4PV2G----5Grid2Inflex------6Cogen2Inflex------7Flex1--------8-Flex2----------9-Grid2Flex----------10-Cogen2Flex----------11-Tro----------------12--Tri----------------13-Tci-----------14-Tco----------------15-Ts---------16-Grid2CRAC------17-Cogen2CRAC----------18G2misc------19-Cogen2misc----20-PV2BESS------21BESS-ON-OF---22PV2Inflex---23PV2Flex-----24PV2CRAC------25PV2Misc------26HeatSell-------C2Grid
ub = [R_bes*ones(1,n)   cgof*R_bes*ones(1,n)       z_            z_      inflex           cgof*inflex         ones(1,n)      ones(1,n)         M * ones(1,n)     cgof*M * ones(1,n)         100* ones(1,n)     100* ones(1,n)       100* ones(1,n)     Tc_o'*-fcop           100* ones(1,n)    1000*ones(1,n)   cgof*1000*ones(1,n)      misc'         cgof*misc'     R_bes*ones(1,n)   ones(1,n)    -solar       -solar      -solar    -solar               z_             z_];           % upper bounds %household bound simply ensure that the house never buys more power than it needs.
lb = [z_                     z_             -R_bes * ones(1,n)   solar     z_                 z_                z_              z_                z_                    z_                  0* ones(1,n)         0* ones(1,n)        0 * ones(1,n)    Tc_o'*-fcop          0 * ones(1,n)       z_                 z_                       z_           z_                    z_            z_           z_           z_           z_         z_         -cogen_heat'     -max_cogen*ones(1,n)];          % lower bound %solar is negative    

A = [   
     t    t    t       z       t   t -cumsum(P1') -cumsum(P2')   t   t     z       z      -fcop*t  -t    z  t  t t  t         t   z      t  t  t  t        z      z;     %14 Constraint - SOE of BESS at time t < Max SOE (Energy balance equation)
    -t   -t   -t      z      -t   -t cumsum(P1')  cumsum(P2')  -t   -t    z       z      fcop*t  t   z  -t   -t -t -t        -t   z       -t   -t -t -t    z      z;     %15 Constraint - SOE of BESS at anytime > Min SOE
    
     e    e   z       z       z     z  z      z      z              z   z       z          z    z  z z z z z                  e    z       z z z z          z      z;               %16   %limits charging rate
   
     
     e    e   e       z       e     e   -P1'  -P2'   e              e    z       z         -fcop*e  -e  z  e  e  e    e       e     z    e  e  e    e        z       z; %instantaneous limits for the BESS
    
     -e   -e   -e      z     -e    -e   P1'  P2'   -e              -e    z       z         fcop*e  e  z  -e  -e  -e    -e     -e     z  -e  -e  -e    -e      z        z;  %instantaneous limits for the BESS


     z    z    z        z       z    z  z      z      z          z   z       z      -fcop*e   -e  z   e   e z z                 z    z   z   z  e   z         z          z;     %CRAC limit (Grid + Cogen)       % -prac_tot + prac_buy < 0   % pcrac = -(tci + Tc_o)*cc / COP; fcop = cc/COP
      z    z   z       z       z    z  -P1'   -P2'    e            e   z       z      z  z   z z  z z z                        z    z    z   e  z    z        z           z;     %Flex limit    (Grid + Cogen)      %23
      
      z    z   z       z       e    e  z   z    z    z   z       z      z  z   z z  z z z                        z    z                   e   z  z   z        z            z;               % Inflex Limit (Grid + Cogen)
      z    z   z       z       z    z  z   z    z    z   z       z      z  z   z z  z e e                        z    z                    z   z  z   e        z           z;               % Miscellanous limit (Grid + Cogen)

   
      e     e   z      z      z  z  z       z     z     z   z    z    z     z  z   z    z    z    z                            e       -R_bes*CE_bes*e    z  z z z     z    z;  %    Limits charging and discharging BESS same time
               
     z    z    -e      z       -e   -e  P1'     P2'   -e   -e   z   z  fcop*e  e z   -e     -e  -e   -e                z       R_bes*CE_bes*e    -e     -e  -e   -e    z     z; 
   
     z    -e   z       z         z   -e  z        z    z   -e    z       z      z  z   z  z     -e  z   -e                    z    z                  z     z  z   z    -e   e;               % Cogen_heat <= Cogen_electricity

     z    e   z       z         z   e  z        z    z   e    z          z      z  z   z  z      e  z   e                    z    z                    z     z  z   z    z   -e;               % Limit cogen power generation

    % z    z   z       z         z   z  z        z    z   z    z       z      z  z   z  z     z  z   z                    z    z                  z     z  z   z    z;               % Cogen_heat <= Cogen_electricity

      ];
      
   
b = [
      DMax_bes+cumsum(inflex)' + cumsum(misc) ;             %14
      DMin_bes-cumsum(inflex)' - cumsum(misc);             %15
    
      R_bes*CE_bes*ones(1,n)';                             %16    
      R_bes*CE_bes*ones(1,n)' + inflex' + misc ;
      R_bes*CE_bes*ones(1,n)' - inflex' - misc  ;

     zeros(1,n)';
     zeros(1,n)';

     inflex';
  
     misc;
    % -solar'%;R_bes*CE_bes*ones(1,n)'
     zeros(1,n)';

      R_bes*CE_bes*ones(1,n)' - inflex' - misc  ; %I changed solar sign from +ve to negative
  
      zeros(1,n)';
      max_cogen*ones(1,n)'
];


        Aeq=[z_ z_ z_ z_ z_ z_ aeq z_ z_   z_  z_   z_   z_   z_ z_ z_ z_ z_ z_     z_  z_  z_ z_     z_  z_  z_  z_;
             z_ z_ z_ z_ z_ z_ z_ aeq z_  z_   z_   z_   z_   z_ z_ z_  z_ z_ z_    z_   z_ z_ z_     z_  z_ z_  z_;
             z z z z z z z z z   z         -e*cr    z    e*c1 z z z  z z z          z    z z z          z    z  z   z; %equation for Tr,o  from (2) %% Mistake no Tco coefficienmt
             z z z z z z z z z   z        z  e*cr   -e*crec  z z z z z z            z    z z z          z    z  z   z;  %equation for Tr,o and Tci from Equation (3)
             z z z z z z -c5*P1' -c5*P2' z z   z  -c4*e   z  z TSS z z z z          z     z z z          z    z  z   z;  %Equation (4)
             z z z z z z c9*P1' c9*P2' z z   e  c8*e   z  z TSS2 z z z z            z     z z z          z    z  z    z;  %equation for Tr,o and Tci from Equation (1)
        
            z     z   z      -e      z   z  z       z     z     z   z    z    z     z  z   z    z    z    z   e    z   e  e  e  e  z   z;  %   PV2G + PV2BESS < Solar 
             ]; %equality constraint for a single TSA(flexible) appliance load. Specifically to constraint the switch binary vector condition. That is for all timesteps sum(x) = 1 e,g x = [0 0 1 0 0 0 0]
        
        beq=[1;
             1;
             Wmisc*ones(1,n)'+ Tc_o*c2; % Equation 2
             Tc_o * c3; % from (4)
             bt' + c5*inflex'; % Equation 4
             bb' - c9*inflex'; % Equation 1

             -solar';
              ]; 
          

[x, fval] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub , options);



%% Post-Processing after Optimization

% TOU, fit, degradation costs
deltat = 1/12*3600;
time=0:deltat/3600:deltat*287/3600;       % time in hours 

%% LOAD PREPROCESSING

%Power Distribution between grid and cogen and the respective power
%consuming components of the DC (BESS, Inflex, Flex, CRAC)
G2BESS = x(1:n,1);
C2BESS = x(n+1:2*n,1);
G2Inflex = x(4*n+1:5*n,1);
C2Inflex = x(5*n+1:6*n,1);
G2flex = x(8*n+1:9*n,1);
C2flex = x(9*n+1:10*n,1);
G2CRAC = x(15*n+1:16*n,1);
C2CRAC = x(16*n+1:17*n,1);
G2misc = x(17*n+1:18*n,1);
C2misc =  x(18*n+1:19*n,1);
PV2BESS =  x(19*n+1:20*n,1);
YY = x(20*n+1:21*n,1);
%% Other  Decision Variables

PV2G = x(3*n+1:4*n,1);
PV2inflex = x(21*n+1:22*n,1);
PV2flex = x(22*n+1:23*n,1);
PV2CRAC= x(23*n+1:24*n,1);
PV2misc = x(24*n+1:25*n,1);
Ch2H = x(25*n+1:26*n,1); %amount of cogen-heat exported or sold to district which reduces electricity cost
C2Grid = x(26*n+1:27*n,1);

BESS2G = x(2*n+1:3*n,1);
flex_1 = P1'*x(6*n+1:7*n,1); %flex1 load schedule
flex_2 = P2'*x(7*n+1:8*n,1); %flex2 load schedule

%% Temperature Related Components
tro = x(10*n+1:11*n,1) ;
tri =  x(11*n+1:12*n,1);
tci = x(12*n+1:13*n,1) ;
tco = x(13*n+1:14*n,1) ; %not used as tco but rather the tco term in crac power equation
tsi =  x(14*n+1:15*n,1);

%% ****** Self-Consumption Post Processing ***********

%% CRAC Self-Consumption
% total CRAC power
pcrac = (tci - Tc_o)*cc / COP; 
%amount of self-consumed power by CRAC unit - might be wrong calculation




%% Load, PV, and BESS Self-Consumption

totalGCPinflex = G2Inflex + C2Inflex + PV2inflex ;% Total Power Bought by Inflex Load from Grid and Cogen

BESS2inflex = inflex' - totalGCPinflex; %inflex load demand that must be covered by onsite generation components (PV and BESS)

totalGCP2flex =  G2flex + C2flex + PV2flex; % Total Power Bought by Inflex Load from Grid and Cogen
%BESS2flex = flex_1  + flex_2; %total flex demand

BESS2flex  = flex_1 +  flex_2  - totalGCP2flex;  % flex load demand that must be covered by onsite generation (PV and BESS)

% TOU, fit, degradation costs
deltat = 1/12*3600;
time=0:deltat/3600:deltat*287/3600;       % time in hours  


%Pv to CRAC load calculations
totalGCP2crac = G2CRAC + C2CRAC + PV2CRAC; 
BESS2CRAC = pcrac  - totalGCP2crac;
BESS2misc = misc - G2misc -C2misc - PV2misc ; % Amount of Misc Load that must be met by self-consumption


%% BESS SYSTEM POST-PROCESSING

G2BESS = G2BESS; %BESS buying from grid (charging)
C2BESS = C2BESS; %BESS charging from Cogen (charging)
BESS2G = BESS2G ; %BESS selling to grid (discharge)
%PV2BESS = pv_left_misc ; %Amount of pv used to charge BESS (charging)
BESS2inflex = BESS2inflex;%infl_left_pv; % Amount of BESS supplied to remaining inflex load  (discharge)
BESS2flex = BESS2flex;% fl_left_pv; % Amount of BESS supplied to remaining flex load  (discharge)
BESS2crac = BESS2CRAC;  % Amount of BESS supplied to remaining crac cooling load  (discharge)
BESS2misc = BESS2misc; %Amount of BESS supplied to remaining misc load  (discharge)
%G2BESS_ini = G2BESS(1) +   bes_ini ; %setting initial SOE

charge_ini =  G2BESS(1) ; %amount BESS charges at time = 1
G2BESS(1) = charge_ini + bes_ini;

SOE = cumsum(G2BESS) + cumsum(C2BESS) + cumsum(PV2BESS) + cumsum(BESS2G) - cumsum(BESS2crac) - cumsum(BESS2misc) - cumsum(BESS2flex) - cumsum(BESS2inflex);

%SOE(1) = SOE(1) + bes_ini;
G2BESS(1) = charge_ini; 

%% PV SYSTEM Post-processing

pv2flex = PV2flex;
pv2inflex = PV2inflex;
pv2crac = PV2CRAC;
PV2BESS = PV2BESS;
PV2G = PV2G;

%% Heating Demand Calculations
C2heat = Ch2H;
heat_left = cogen_heat + C2heat;

%% ==== NEW - POST-PROCESSING - END ====

% TOU, fit, degradation costs
deltat = 1/12*3600;
time=0:deltat/3600:deltat*287/3600;       % time in hours  





figure(101)

hold on 

BESS_profile = [G2BESS PV2BESS BESS2G C2BESS -BESS2flex -BESS2inflex -BESS2crac -BESS2misc];% BESS2Inflex];% C2flex C2CRAC  C2Heat]; 
YY = x(20*n+1:21*n,1);


l = bar(time, BESS_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %inf_left
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(5),'FaceColor', [0.3010 0.7450 0.9330],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(6),'FaceColor', [0.6350 0.0780 0.1840],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(7),'FaceColor', [0.4660 0.6740 0.1880],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(8),'FaceColor', [0.2660 0.9740 0.1880],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('BESS Power Distribution')

hold on

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
%ylim([-150 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' ON-OFF',  'FontSize',24,'FontWeight', 'bold')

ylim([-10 2])
%plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
%plot(time,price, LineWidth=3, Color='b')

%plot(time,FiT*ones(288,1), LineWidth=3)

plot(time, YY ,LineWidth=3)

lgd = legend('G2BESS', 'PV2BESS', 'BESS2G', 'C2BESS','BESS2Flex','BESS2Inflex', 'BESS2CRAC', 'BESS2Misc', 'Binary',  'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on



%% == FIGURES MONITORING THE DIFFERENT COMPONENTS

% TOU, fit, degradation costs
deltat = 1/12*3600;
time=0:deltat/3600:deltat*287/3600;       % time in hours  



%% PV Analysis

figure(2)

hold on 

pv_profile = [pv2flex pv2inflex pv2crac PV2BESS -PV2G PV2misc];% BESS2Inflex];% C2flex C2CRAC  C2Heat]; 



l = bar(time, pv_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %inf_left
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(5),'FaceColor', [0.3010 0.7450 0.9330],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(6),'FaceColor', [0.6350 0.0780 0.1840],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('PV Power Distribution')

hold on
plot(time,-solar,"LineStyle","--" ,LineWidth=1.5 , Color='red')

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
%ylim([-150 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')



plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)

%plot(time, SOE)

lgd = legend('PV2Flex', 'PV2Inflex', 'PV2CRAC', 'PV2BESS','PV2G', 'PV2Misc', 'PV profile','Buying Price', 'Selling Price',  'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on

%% BESS Analysis Figure

figure(1)

hold on 

BESS_profile = [G2BESS PV2BESS BESS2G C2BESS -BESS2flex -BESS2inflex -BESS2crac -BESS2misc];% BESS2Inflex];% C2flex C2CRAC  C2Heat]; 



l = bar(time, BESS_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %inf_left
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(5),'FaceColor', [0.3010 0.7450 0.9330],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(6),'FaceColor', [0.6350 0.0780 0.1840],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(7),'FaceColor', [0.4660 0.6740 0.1880],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(8),'FaceColor', [0.2660 0.9740 0.1880],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('BESS Power Distribution')

hold on

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
ylim([-150 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh) - SOE',  'FontSize',24,'FontWeight', 'bold')


%plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)

plot(time, SOE ,LineWidth=3)

lgd = legend('G2BESS', 'PV2BESS', 'BESS2G', 'C2BESS','BESS2Flex','BESS2Inflex', 'BESS2CRAC', 'BESS2Misc', 'Buying Price', 'Selling Price', 'SOE',  'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on

%% COGEN Analysis Figure


%% Heating Demand Calculations
C2heat = Ch2H;
heat_left = cogen_heat + C2heat;


figure(1112)

hold on 


tiledlayout(2,1)

% Top plot
nexttile

cogen_heat_profile = [C2BESS C2flex C2Inflex C2CRAC C2misc -C2Grid];% BESS2Inflex];% C2flex C2CRAC  C2Heat]; 



l = bar(time, cogen_heat_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %inf_left
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(5),'FaceColor', [0.3010 0.7450 0.9330],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
set(l(6),'FaceColor', [0.6350 0.0780 0.1840],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
%set(l(7),'FaceColor', 'white', 'EdgeColor', [0.4660 0.6740 0.1880], 'LineWidth', 0.5);  %flex1_grid
%set(l(8),'FaceColor', [0.2660 0.9740 0.1880],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('Congeneration Power Distribution')

hold on

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
%ylim([-150 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh) - SOE',  'FontSize',24,'FontWeight', 'bold')


plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
%plot(time,cogen_heat, LineWidth=3, Color='b')

plot(time,cogen_sell, LineWidth=3)

%plot(time, SOE ,LineWidth=3)

lgd = legend('C2BESS','C2Flex','C2Inflex', 'C2CRAC', 'C2Misc', 'C2Grid', ' Heat Selling Price', 'Cogen_Selling Price',  'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on

nexttile
hold on
co_profile = [-C2heat];% BESS2Inflex];% C2flex C2CRAC  C2Heat]; 



l = bar(time, co_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
%set(l(2),'FaceColor','white' , 'EdgeColor' , [0 0.4470 0.7410]);   %C2Inflex


plot(time,cogen_heat, LineWidth=3)

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('Congeneration Heat Distribution')

hold on

xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
%ylim([-150 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
%plot(time,cogen_heat, LineWidth=3, Color='b')



%plot(time, SOE ,LineWidth=3)

lgd = legend('Cogen Export', 'Heating Demand' , 'Heat Selling Price',  'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on


%% Load Analysis Figure

figure(122)

hold on 

inflex_load_profile = [G2Inflex pv2inflex BESS2inflex C2Inflex];% C2flex C2CRAC  C2Heat]; 



tiledlayout(4,1)

% Top plot
nexttile


l = bar(time, inflex_load_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %inf_left
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('Inflex Load Power Demand Distribution')

hold on

plot(time,inflex ,LineWidth=1.5 , Color='cyan')


xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
ylim([0 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


%plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)


lgd = legend('G2Inflex', 'PV2Flex', 'BESS2Inflex', 'C2Inflex', 'Inflex Profile', 'Buying Price', 'Selling Price', 'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on


% Middle plot
nexttile

flex_load_profile =[G2flex  pv2flex BESS2flex C2flex];
l = bar(time, flex_load_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %inf_left
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('Inflex Load Power Demand Distribution')

hold on

%plot(time,-cogen_heat ,LineWidth=1.5 , Color='cyan')


xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
ylim([0 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


%plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)


lgd = legend('G2flex','PV2flex', 'BESS2Flex', 'C2Flex','Buying Price', 'Selling Price', 'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on

title('Flex Load Power Demand Distribution')


% Bottom plot
nexttile

crac_load_profile =[G2CRAC  pv2crac BESS2crac C2CRAC];
l = bar(time, crac_load_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %pv2crac
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none');  %BESS2crac
xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('CRAC Load Power Demand Distribution')

hold on

plot(time,pcrac ,LineWidth=1.5 , Color='cyan')


xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
ylim([0 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


%plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)


lgd = legend('G2CRAC','PV2CRAC', 'BESS2CRAC', 'C2Flex', 'CRAC Profile', 'Buying Price', 'Selling Price', 'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on

title('CRAC Load Power Demand Distribution')




% Fourth plot
nexttile

misc_load_profile =[G2misc  PV2misc BESS2misc C2misc];
l = bar(time, misc_load_profile, 'stacked');

set(l(1),'FaceColor',[0 0.4470 0.7410] , 'EdgeColor' , 'none');   % 
set(l(2),'FaceColor',[0.8500 0.3250 0.0980] , 'EdgeColor' , 'none');   %C2Inflex
set(l(3), 'FaceColor', [0.9290 0.6940 0.1250], 'EdgeColor' , 'none'); %inf_left
set(l(4),'FaceColor', [0.4940 0.1840 0.5560],'EdgeColor', 'none', 'LineWidth', 0.5);  %flex1_grid
xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
title('Inflex Load Power Demand Distribution')

hold on

plot(time,misc ,LineWidth=1.5 , Color='cyan')


xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 
xticks(0: 6: 24);
ylim([0 150])
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')

yyaxis right
hold on
ylabel(' Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


%plot(time,heat_pr,"LineStyle","--" ,LineWidth=1.5 , Color='red')
plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)


lgd = legend('G2Misc','PV2Misc', 'BESS2Misc', 'C2Misc', 'Misc Profile','Buying Price', 'Selling Price', 'Fontsize', 18);
lgd.NumColumns = 5;

set(lgd,'Location','north') %'eastoutside'
grid on

title('Misc Load Power Demand Distribution')










%% *** Compute Proportion Of Power Bought From Either Grid or Cogeneration
%%By Each Flexible Load

load_self_con = pv2flex + BESS2flex ; %Flexible load demand met by onsite power (Self Consumption)

% Initialize weight vectors
weight_1 = zeros(1, 288);
weight_2 = zeros(1, 288);

% Compute weights and proportion vectors
for i = 1:288
    weight_sum = flex_1(i) + flex_2(i);
    if weight_sum > 0
        weight_1(i) = flex_1(i) / weight_sum;
        weight_2(i) = flex_2(i) / weight_sum;
    else
        weight_1(i) = 0;
        weight_2(i) = 0;
    end
end

flex1_self = abs(weight_1' .* load_self_con); %amount of flexible 1 that's self-consumed.
flex2_self = abs(weight_2' .* load_self_con); %amount of flexible 2 that's self-consumed.

flex1_grid = flex_1 - flex1_self ; % Amount of flexible 1 load demand bought from grid
flex2_grid = flex_2 - flex2_self ; % Amount of flexible 2 load demand bought from grid


%data = [misc x(3*n+1:4*n,1) scons flex1_grid flex1_self flex2_grid flex2_self x(1:n,1)  x(n+1:n*2,1) pcrac_buy pcrac_self];  

% TOU, fit, degradation costs
deltat = 1/12*3600;
time=0:deltat/3600:deltat*287/3600;       % time in hours  



if cgof == 0

figure(2222)

 hold on

%profile_load = [misc x(3*n+1:4*n,1) scons P2'*x(5*n+1:6*n,1) P1'*x(4*n+1:5*n,1) x(6*n+1:7*n,1) x(1:n,1)  x(n+1:n*2,1)]; 
%data = [misc x(3*n+1:4*n,1) scons P2'*x(5*n+1:6*n,1) P1'*x(4*n+1:5*n,1) x(6*n+1:7*n,1) x(1:n,1)  x(n+1:n*2,1) pcrac];  

%data = [misc x(3*n+1:4*n,1) scons flex1_grid flex1_self flex2_grid flex2_self x(1:n,1)  x(n+1:n*2,1) pcrac_buy pcrac_self];  

inflex_self = BESS2inflex + PV2inflex;
misc_self = BESS2misc + PV2misc;
CRAC_self = BESS2CRAC + PV2CRAC;

data = [G2Inflex inflex_self flex1_grid flex1_self flex2_grid flex2_self G2misc misc_self G2CRAC CRAC_self PV2G  G2BESS BESS2G];  



h = bar(time, data, 'stacked');

set(h(1),'FaceColor',[0.6350 0.0780 0.1840] , 'EdgeColor' , 'none');   % set color of 'misc' series to gray
%set(h(2),'FaceColor',[0.10 0.0780 0.1840] , 'EdgeColor' , 'none');   % set color of 'misc' series to gray
set(h(2), 'FaceColor', 'white' , 'EdgeColor' , [0.6350 0.0780 0.1840], 'LineWidth', 0.5); %inflex     % set color of 'x(3*n+1:4*n,1)' series to blue
set(h(3), 'FaceColor', [0 0 1], 'LineWidth', 0.5 , 'EdgeColor' , 'none');  %flex1 buy      % set color of 'P2''*x(5*n+1:6*n,1)' series to yellow
set(h(4), 'FaceColor', 'white' , 'EdgeColor' , [0 0 1], 'LineWidth', 0.5); %inflex     % set color of 'x(3*n+1:4*n,1)' series to blue
set(h(5), 'FaceColor', [0.4660 0.6740 0.1880],  'EdgeColor' , 'none', 'LineWidth', 0.5);     %flex2 buy       % set color of 'P1''*x(4*n+1:5*n,1)' series to green
set(h(6), 'FaceColor', 'white' , 'EdgeColor' , [0.4660 0.6740 0.1880], 'LineWidth', 0.5); %inflex     % set color of 'x(3*n+1:4*n,1)' series to blue
%set(h(8),'FaceColor',[0.6 0.75 0.6 ], 'EdgeColor' , 'none', 'LineWidth', 0.5);  %flex2 self      % set color of 'scons' series to magenta
%above is inflex and flex loads

set(h(7),'FaceColor',[0.4940 0.1840 0.5560] ,  'EdgeColor' , 'none');           % set color of 'x(6*n+1:7*n,1)' series to red
%set(h(10),'FaceColor',[0 0.8 0.66] ,  'EdgeColor' , 'none');          % set color of 'x(1:n,1)' series to cyan
set(h(8),'FaceColor','white','EdgeColor',[0.4940 0.1840 0.5560], 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(9),'FaceColor',[0.9290 0.6940 0.1250],'EdgeColor','none');     % set color of 'x(n+1:n*2,1)' series to orange
%set(h(13),'FaceColor',[0.1 0.840 0.5560] ,  'EdgeColor' , 'none','LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(10),'FaceColor','white','EdgeColor',[0.9290 0.6940 0.1250], 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(11),'FaceColor',[0 1 1],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(12),'FaceColor',[0 0.4470 0.7410],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(13),'FaceColor',[1 0 1],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
%set(h(14),'FaceColor',[0.1 0.840 0.5560],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange

%}

%bar(time,[misc])
hold on
%h2 = bar(time,[x(2*n+1:3*n,1) ] ,'stacked');
%set(h2(1),'FaceColor','black');   % set color of 'misc' series to gray
%title('Daily load profile')
grid on
plot(time,inflex,  LineWidth=3)
%plot(time, pcrac ,"LineStyle","--")
plot(time,solar, LineWidth=3)

%plot(time,soc,"LineStyle","--" ,LineWidth=3 , Color='yellow')
%ylim([-8 8])
xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 

ylim([-200 300])
xticks(0: 2: 24);
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')
yyaxis right
hold on
ylabel('Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)
%plot(time,cogen_price,"LineStyle","--" ,LineWidth=3 , Color='g')
%plot(time,rtp_price,"LineStyle","--" ,LineWidth=3 , Color='cyan')


lgd = legend('G2Inflex', 'Inflex Self', 'G2Flex-1', 'Flex-1 Self', 'G2Flex-2','Flex-2 Self',...
    'G2Misc', 'Misc Self','G2CRAC', 'CRAC Self', 'PV2Grid',  'BESS Buy', 'BESS Sell',...
    'Inflex Profile','PV Production',  'Buying Price',  'Selling Price',  'Fontsize', 20);
%legend('Misc Load', 'Inflex Load (Buy)', 'Inflex Load (Self)', 'Flex Load_1 (Self)','Flex Load_2 (Self)','Flex Load (Buy)', 'BESS Charge','BESS Discharge'...
%    ,'CRAC Power', 'PV2Grid', 'Inflex Load', 'PV Production', 'Buying Price',  'Selling Price', 'Fontsize', 24)

%lgd = legend('Inflex Load (Buy)', 'Inflex Load (Self)', 'Flex Load-1 (Buy)', 'Flex Load-1 (Self)', 'Flex Load-2 (Buy)', 'Flex Load-2 (Self)', 'BESS Charge','BESS Discharge'...
 %  ,'Cooling Power Buy', 'Cooling Power Self', 'PV2Grid', 'Inflex Load', 'PV Production', 'Buying Price',  'Selling Price', 'Fontsize', 24);
lgd.NumColumns = 6;

set(lgd,'Location','northeast') %'eastoutside'
ylim([0 2.5])
%yticks(0: 0.1: 1);
yticks(0: 0.25: 1);

ax=gca;
ax.FontSize = 24;

else

figure(2222)

 hold on

%profile_load = [misc x(3*n+1:4*n,1) scons P2'*x(5*n+1:6*n,1) P1'*x(4*n+1:5*n,1) x(6*n+1:7*n,1) x(1:n,1)  x(n+1:n*2,1)]; 
%data = [misc x(3*n+1:4*n,1) scons P2'*x(5*n+1:6*n,1) P1'*x(4*n+1:5*n,1) x(6*n+1:7*n,1) x(1:n,1)  x(n+1:n*2,1) pcrac];  

%data = [misc x(3*n+1:4*n,1) scons flex1_grid flex1_self flex2_grid flex2_self x(1:n,1)  x(n+1:n*2,1) pcrac_buy pcrac_self];  

inflex_self = BESS2inflex + PV2inflex;
misc_self = BESS2misc + PV2misc;
CRAC_self = BESS2CRAC + PV2CRAC;

data = [G2Inflex C2Inflex inflex_self flex1_grid flex1_self flex2_grid flex2_self C2flex G2misc C2misc misc_self G2CRAC  C2CRAC CRAC_self PV2G  G2BESS BESS2G C2BESS];  



h = bar(time, data, 'stacked');

set(h(1),'FaceColor',[0.6350 0.0780 0.1840] , 'EdgeColor' , 'none');   % set color of 'misc' series to gray
set(h(2),'FaceColor',[0.10 0.0780 0.1840] , 'EdgeColor' , 'none');   % set color of 'misc' series to gray
set(h(3), 'FaceColor', 'white' , 'EdgeColor' , [0.6350 0.0780 0.1840], 'LineWidth', 0.5); %inflex     % set color of 'x(3*n+1:4*n,1)' series to blue
set(h(4), 'FaceColor', [0 0 1], 'LineWidth', 0.5 , 'EdgeColor' , 'none');  %flex1 buy      % set color of 'P2''*x(5*n+1:6*n,1)' series to yellow
set(h(5), 'FaceColor', 'white' , 'EdgeColor' , [0 0 1], 'LineWidth', 0.5); %inflex     % set color of 'x(3*n+1:4*n,1)' series to blue
set(h(6), 'FaceColor', [0 0.75 0 ],  'EdgeColor' , 'none', 'LineWidth', 0.5);     %flex2 buy       % set color of 'P1''*x(4*n+1:5*n,1)' series to green
set(h(7), 'FaceColor', 'white' , 'EdgeColor' , [0 0.75 0], 'LineWidth', 0.5); %inflex     % set color of 'x(3*n+1:4*n,1)' series to blue
set(h(8),'FaceColor',[0.6 0.75 0.6 ], 'EdgeColor' , 'none', 'LineWidth', 0.5);  %flex2 self      % set color of 'scons' series to magenta
%above is inflex and flex loads

set(h(9),'FaceColor',[0.4940 0.1840 0.5560] ,  'EdgeColor' , 'none');           % set color of 'x(6*n+1:7*n,1)' series to red
set(h(10),'FaceColor',[0 0.8 0.66] ,  'EdgeColor' , 'none');          % set color of 'x(1:n,1)' series to cyan
set(h(11),'FaceColor','white','EdgeColor',[0.4940 0.1840 0.5560], 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(12),'FaceColor',[0.9290 0.6940 0.1250],'EdgeColor','none');     % set color of 'x(n+1:n*2,1)' series to orange
set(h(13),'FaceColor',[0.1 0.840 0.5560] ,  'EdgeColor' , 'none','LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(14),'FaceColor','white','EdgeColor',[0.9290 0.6940 0.1250], 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(15),'FaceColor',[0 1 1],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(16),'FaceColor',[1 0 0],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
set(h(17),'FaceColor',[1 0 1],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange


set(h(18),'FaceColor',[0.1840 0.4940 0.5560],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange
%set(h(19),'FaceColor',[0.1250 0.6940 0.9290],'EdgeColor','none', 'LineWidth', 0.5);     % set color of 'x(n+1:n*2,1)' series to orange


%}

%bar(time,[misc])
hold on
%h2 = bar(time,[x(2*n+1:3*n,1) ] ,'stacked');
%set(h2(1),'FaceColor','black');   % set color of 'misc' series to gray
%title('Daily load profile')
grid on
plot(time,inflex,  LineWidth=3)
%plot(time, pcrac ,"LineStyle","--")
plot(time,solar, LineWidth=3)

%plot(time,soc,"LineStyle","--" ,LineWidth=3 , Color='yellow')
%ylim([-8 8])
xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 

ylim([-100 400])
xticks(0: 6: 24);
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')
yyaxis right
hold on
ylabel('Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)
plot(time,cogen_price,"LineStyle","--" ,LineWidth=3 , Color='g')
plot(time,cogen_sell,"LineStyle","--" ,LineWidth=3)

plot(time,cogen_heat ,"LineStyle","--" ,LineWidth=3 , Color='cyan')




lgd = legend('G2Inflex','Cogen2Inflex', 'Inflex Self', 'G2Flex-1', 'Flex-1 Self', 'G2Flex-2','Flex-2 Self', 'Cogen2Flex',...
    'G2Misc','Cogen2Misc', 'Misc Self','G2CRAC' , 'Cogen2CRAC', 'CRAC Self', 'PV2Grid',  'BESS Buy', 'BESS Sell','C2BESS',...
    'Inflex Profile','PV Production',  'Buying Price',  'Selling Price', 'Heating Buying Price','Cogen Elec Selling Price', 'Heat Demand Profile',  'Fontsize', 18);
%legend('Misc Load', 'Inflex Load (Buy)', 'Inflex Load (Self)', 'Flex Load_1 (Self)','Flex Load_2 (Self)','Flex Load (Buy)', 'BESS Charge','BESS Discharge'...
%    ,'CRAC Power', 'PV2Grid', 'Inflex Load', 'PV Production', 'Buying Price',  'Selling Price', 'Fontsize', 24)

%lgd = legend('Inflex Load (Buy)', 'Inflex Load (Self)', 'Flex Load-1 (Buy)', 'Flex Load-1 (Self)', 'Flex Load-2 (Buy)', 'Flex Load-2 (Self)', 'BESS Charge','BESS Discharge'...
 %  ,'Cooling Power Buy', 'Cooling Power Self', 'PV2Grid', 'Inflex Load', 'PV Production', 'Buying Price',  'Selling Price', 'Fontsize', 24);
lgd.NumColumns = 3;

set(lgd,'Location','north') %'eastoutside'
ylim([0 0.5])
%yticks(0: 0.1: 1);
yticks(0: 0.05: 0.5);

ax=gca;
ax.FontSize = 24;

end
%ylim([0 0.4])

% x(1:n,1)  x(n+1:n*2,1) 'Batarya Şarj','Batarya Deşarj',
% alpha(0.5)




figure(11)

bess_profile =[G2BESS  C2BESS]; 
inflex_profile = [G2Inflex  C2Inflex];
flex_profile = [G2flex C2flex];
CRAC_profile = [G2CRAC C2CRAC];
all_profile = [G2BESS  C2BESS G2Inflex  C2Inflex G2flex C2flex G2CRAC C2CRAC];
 
hold on
h = bar(time,all_profile,'stacked','DisplayName','ALL Profile');
set(h(1),'FaceColor',[0.6350 0.0780 1] , 'EdgeColor' , 'none'); 

title('Grid and Cogeneration Power Distribution')
hold on
grid on

xlabel('Time (hour)','FontSize',28,'FontWeight', 'bold') 
yyaxis left
ylabel('Power (kW)' ,'FontSize',28,'FontWeight', 'bold')
yyaxis right
ylabel('Elecrticity Price ($/kWh)', 'FontSize',28,'FontWeight', 'bold')
plot(time,price,"LineStyle","--", LineWidth=3)
plot(time,cogen_price,"LineStyle","--" ,LineWidth=3 , Color='g')
plot(time,FiT*ones(288,1),"LineStyle","--" ,LineWidth=3, Color='b')


legend('Grid2BESS','Cogen2BESS','Grid2Inflex','Cogen2Inflex', 'Grid2Flex',...
    'Cogen2Flex', 'Grid2CRAC', 'Cogen2CRAC', 'Grid Buying Price',  'Cogen Buying Price', 'Selling Price','Fontsize', 24)
set(legend,'Location','northeast')
yyaxis right


% TOU, fit, degradation costs
deltat = 1/12*3600;
time=0:deltat/3600:deltat*287/3600;       % time in hours  





figure(3)

hold on
plot(time,tro)
plot(time,tri)
plot(time,tci)
plot(time,tsi)
plot(time,Tc_o)
%plot(time,tro_max)

title('Temperature profile')
grid on

xlabel('Zaman (saat)' ,'FontSize', 14,'FontWeight', 'bold') 
yyaxis left
ylabel('Sıcaklık (Celsius)'  ,'FontSize',14,'FontWeight', 'bold')
yyaxis right
ylabel('Elektrık Fıyatı ($/kWh)' ,'FontSize',14,'FontWeight', 'bold')
plot(time,price )
plot(time,FiT*ones(288,1))
legend('Kabin Çıkışı (Tro)','Kabin Girişi (Tri)','CRAC Girişi (Tci)','Sunucu Kütlesi(Ts)', 'CRAC Çıkışı (Tco)'...
    ,'Alış Fıyatı', 'Satış Fıyatı' ,'FontSize',14)
set(legend,'Location','northeast')
yyaxis right



tou_price=[T3*ones(1,12) T3*ones(1,12) T3*ones(1,12)...
    T3*ones(1,12) T3*ones(1,12) T3*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T1*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T1*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T1*ones(1,12)...
    T1*ones(1,12) T1*ones(1,12) T2*ones(1,12)...         % Each hour represents 12 price time slots of 5 minutes interval.
    T2*ones(1,12) T2*ones(1,12) T2*ones(1,12)...        %price is vector has dimension [288 1] which represents price for 24 hrs expressed every 5 minutes interval (24 *60 / 5 = 288)
    T2*ones(1,12) T3*ones(1,12) T3*ones(1,12)]';        % electricity buying price



rtp_price=1.03*[0.11*ones(1,12) 0.08*ones(1,12) 0.065*ones(1,12)...
    0.05*ones(1,12) 0.065*ones(1,12) 0.11*ones(1,12)...
    0.23*ones(1,12) 0.20*ones(1,12) 0.16*ones(1,12)...
    0.14*ones(1,12) 0.16*ones(1,12) 0.21*ones(1,12)...
    0.18*ones(1,12) 0.17*ones(1,12) 0.16*ones(1,12)...
    0.18*ones(1,12) 0.20*ones(1,12) 0.24*ones(1,12)...         % Each hour represents 12 price time slots of 5 minutes interval.
    0.31*ones(1,12) 0.37*ones(1,12) 0.32*ones(1,12)...        %price is vector has dimension [288 1] which represents price for 24 hrs expressed every 5 minutes interval (24 *60 / 5 = 288)
    0.24*ones(1,12) 0.16*ones(1,12) 0.12*ones(1,12)]';        % electricity buying price

Tcpp3 = T3 - 0.1*T3;
Tcpp1 = T1 - 0.1*T1;
Tcpp2 = T2 - 0.1*T2;
Tcpp = 3*T2;

cpp_price=[Tcpp3*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)...
    Tcpp3*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp1*ones(1,12)...
    Tcpp1*ones(1,12) Tcpp1*ones(1,12) Tcpp*ones(1,12)...         % Each hour represents 12 price time slots of 5 minutes interval.
    Tcpp*ones(1,12) Tcpp*ones(1,12) Tcpp*ones(1,12)...        %price is vector has dimension [288 1] which represents price for 24 hrs expressed every 5 minutes interval (24 *60 / 5 = 288)
    Tcpp*ones(1,12) Tcpp3*ones(1,12) Tcpp3*ones(1,12)]';        % electricity buying price


frp_price = 0.185*ones(1,288)';


figure(322)

hold on
plot(time,frp_price ,LineWidth=3)
plot(time,tou_price , LineWidth=3)
plot(time,rtp_price , LineWidth=3)
plot(time,cpp_price , LineWidth=3)

%plot(time,tro_max)

%title('')
grid on

xlabel('Time (hour)' ,'FontSize', 24,'FontWeight', 'bold') 

ylabel('Electricity Price ($/kWh)'  ,'FontSize',24,'FontWeight', 'bold')

legend('Flat Rate Pricing','Time of Use Pricing','Real Time Pricing','Critical Peak Pricing','FontSize',24)
set(legend,'Location','northeast')
xticks(0: 6: 24);
ax = gca;
ax.FontSize = 20;





%% Post System Capacity Check After Load Scheduling
disp("*** POST OPTIMIZATION SYSTEM CAPACITY VIOLATION CHECK AFTER LOAD SCHEDULING ***")



 %% Compute Total Amount of Electricity Bill Paid by DC
disardan = G2flex + G2Inflex + G2BESS + G2CRAC + G2misc;
energy2G = PV2G +  BESS2G; %pv2grid, BESS2grid
energypaid = sum(disardan.*pr_buy)+sum(energy2G.*pr_sell); 
fprintf('Total Cost of Electricity = %f \n', energypaid);

%% Monitoring Electricity Values
%{
fprintf('Total Flex load Energy bought from Grid (kWh) = %f \n', sum(G2flex.*pr_buy));
fprintf('Total Flex load Self-consumption (kWh) = %f \n', sum(fl_left));
fprintf('Total Flex Percentage (kWh) = %f \n', 100*sum(fl_left)/sum(flex_1 + flex_2));


fprintf('Total inflex Energy Bought from grid(kWh) = %f \n', sum(G2inflex.*pr_buy));
fprintf('Total inflex load Self-consumption (kWh) = %f \n', sum(inf_left));
fprintf('Total inflex Percentage (kWh) = %f \n', 100*sum(inf_left)/sum(inflex));


fprintf('Total Cooling Power Bought from grid(kWh) = %f \n', sum(G2CRAC.*pr_buy));
fprintf('Total Cooling load Self-consumption (kWh) = %f \n', sum(pcrac_self));



fprintf('Total Misc Energy Bought from grid(kWh) = %f \n', sum(G2misc.*pr_buy));
fprintf('Total Misc load Self-consumption (kWh) = %f \n', sum(misc_left));


fprintf('Total BESS bought from grid (kWh) = %f \n', sum(G2BESS.*pr_buy));
fprintf('Total BESS sold to Grid(kWh) = %f \n', sum(BESS2G));


fprintf('Total PV sold to Grid(kWh) = %f \n', -sum(PV2G));
fprintf('Total PV Self-Consumed Grid(kWh) = %f \n', sum(pv_left));
%}
fprintf('Average Flex = %f \n', mean(flex_1(1:72)));
fprintf('Average Inflex = %f \n', mean(inflex));

fprintf('Min Inflex = %f \n', min(inflex));

fprintf('Max Inflex = %f \n', max(inflex));


%{


% figure(1)

bess_profile =[G2BESS  C2BESS]; 
inflex_profile = [G2Inflex  C2Inflex];
flex_profile = [G2flex C2flex];
CRAC_profile = [G2CRAC C2CRAC];

 
hold on
h = bar(time,bess_profile,'stacked','DisplayName','BESS Profile');
set(h(1),'FaceColor',[0.6350 0.0780 0.1840] , 'EdgeColor' , 'none'); 
set(h(2), 'FaceColor', [ 1 0 1] , 'EdgeColor' , 'none');
hold on
h1 = bar(time,inflex_profile,'stacked','DisplayName','Inflex Profile');
set(h1(1),'FaceColor',[0.6350 0.0780 0.1840] , 'EdgeColor' , 'none'); 
set(h1(2), 'FaceColor', [ 1 0 1] , 'EdgeColor' , 'none');


h2 = bar(time,flex_profile,'stacked','DisplayName','Flex Profile');

set(h2(1),'FaceColor',[0.6350 0.0780 0.1840] , 'EdgeColor' , 'none'); 
set(h2(2), 'FaceColor', [ 1 0 1] , 'EdgeColor' , 'none');

hold on
h3 = bar(time,CRAC_profile,'stacked','DisplayName','CRAC Profile');

set(h3(1),'FaceColor',[0.6350 0.0780 0.1840] , 'EdgeColor' , 'none'); 
set(h3(2), 'FaceColor', [ 1 0 1] , 'EdgeColor' , 'none');
title('Grid and Cogeneration Power Distribution')
hold on
grid on

xlabel('Time (hour)','FontSize',28,'FontWeight', 'bold') 
yyaxis left
ylabel('Power (kW)' ,'FontSize',28,'FontWeight', 'bold')
yyaxis right
ylabel('Elecrticity Price ($/kWh)', 'FontSize',28,'FontWeight', 'bold')
plot(time,price,"LineStyle","--", LineWidth=3)
plot(time,cogen_price,"LineStyle","--" ,LineWidth=3 , Color='g')
plot(time,FiT*ones(288,1),"LineStyle","--" ,LineWidth=3, Color='b')


legend('Grid2BESS','Cogen2BESS','Grid2Inflex','Cogen2Inflex', 'Grid2Flex',...
    'Cogen2Flex', 'Grid2CRAC', 'Cogen2CRAC', 'Grid Buying Price',  'Cogen Buying Price', 'Selling Price','Fontsize', 24)
set(legend,'Location','northeast')
yyaxis right





figure(2222)

hold on


data = [misc x(3*n+1:4*n,1) scons flex1_grid flex1_self flex2_grid flex2_self x(1:n,1)  x(n+1:n*2,1) pcrac_buy pcrac_self];  

h = bar(time, data, 'stacked');

set(h(1),'FaceColor',[0.6350 0.0780 0.1840] , 'EdgeColor' , 'none');   % set color of 'misc' series to gray
set(h(2), 'FaceColor', [ 1 0 1] , 'EdgeColor' , 'none'); %inflex     % set color of 'x(3*n+1:4*n,1)' series to blue
set(h(3),'FaceColor','white','EdgeColor', [1 0 1], 'LineWidth', 0.5);  %inflex  self    % set color of 'scons' series to magenta
set(h(4), 'FaceColor', [0 0 1], 'LineWidth', 0.5 , 'EdgeColor' , 'none');  %flex1 buy      % set color of 'P2''*x(5*n+1:6*n,1)' series to yellow
set(h(5),'FaceColor',[0.75 0.75 1], 'EdgeColor' , 'none', 'LineWidth', 0.5);    %flex1 self   % set color of 'scons' series to magenta
set(h(6), 'FaceColor', [0 0.75 0 ],  'EdgeColor' , 'none', 'LineWidth', 0.5);     %flex2 buy       % set color of 'P1''*x(4*n+1:5*n,1)' series to green
set(h(7),'FaceColor',[0.6 0.75 0.6 ], 'EdgeColor' , 'none', 'LineWidth', 0.5);  %flex2 self      % set color of 'scons' series to magenta
set(h(8),'FaceColor',[0.4940 0.1840 0.5560] ,  'EdgeColor' , 'none');           % set color of 'x(6*n+1:7*n,1)' series to red
set(h(9),'FaceColor',[1 1 0] ,  'EdgeColor' , 'none');          % set color of 'x(1:n,1)' series to cyan
set(h(10));       % set color of 'x(n+1:n*2,1)' series to orange
set(h(11),'FaceColor','white','EdgeColor', [0 1 1], 'LineWidth', 0.5);

%bar(time,[misc])
hold on
h2 = bar(time,[x(2*n+1:3*n,1) ] ,'stacked');
set(h2(1),'FaceColor','black');   % set color of 'misc' series to gray
%title('Daily load profile')
hold on
grid on
plot(time,minflex, LineWidth=3)
%plot(time, pcrac ,"LineStyle","--")
plot(time,solar, LineWidth=3)


%ylim([-8 8])
xlabel('Time (hour)' ,'FontSize',28,'FontWeight', 'bold') 

ylim([-100 400])
xticks(0: 6: 24);
yyaxis left
ylabel('Power (kW)', 'FontSize',28,'FontWeight', 'bold')
yyaxis right
hold on
ylabel('Electricity Price ($/kWh)',  'FontSize',24,'FontWeight', 'bold')


plot(time,price, LineWidth=3, Color='b')

plot(time,FiT*ones(288,1), LineWidth=3)

%legend('Misc Load', 'Inflex Load (Buy)', 'Inflex Load (Self)', 'Flex Load_1 (Self)','Flex Load_2 (Self)','Flex Load (Buy)', 'BESS Charge','BESS Discharge'...
%    ,'CRAC Power', 'PV2Grid', 'Inflex Load', 'PV Production', 'Buying Price',  'Selling Price', 'Fontsize', 24)

lgd = legend('Misc Load', 'Inflex Load (Buy)', 'Inflex Load (Self)', 'Flex Load-1 (Buy)', 'Flex Load-1 (Self)', 'Flex Load-2 (Buy)', 'Flex Load-2 (Self)', 'BESS Charge','BESS Discharge'...
   ,'Cooling Power Buy', 'Cooling Power Self', 'PV2Grid', 'Inflex Load', 'PV Production', 'Buying Price',  'Selling Price', 'Fontsize', 24);
lgd.NumColumns = 4;

set(lgd,'Location','north') %'eastoutside'
ylim([0 1])
yticks(0: 0.1: 1);

ax=gca;
ax.FontSize = 24;

%}


%% Function to limit the Execution Time of Loads to End of Day
function E = limit_execution(flex , P1, start)
    %flex = flexible
    n = length(flex);
    
    %****Ensure that loads are executed before the day ends
    non_zero_idx = 0; %index of the last occuring non-zero load in the profile
    
    inf = 1000; %infinite load to restrict execution at certain time steps
    
    %loop to get index of last non-zero element in profile
    for i = 1 : n 
        if flex(i) ~= 0
            non_zero_idx = i ;
        end
    end
    
    % maximum number of shifts from the last non-zero element till the end of
    % the day
    s = n - non_zero_idx;
    
    %user enter start time of TSA
    start_time = start; %load can only be executed only after the 3rd timestep.
    %user enter stop time of TSA
    %end_time = 6; 
    
    if n - start_time < non_zero_idx 
        disp('Time is not sufficient to finish execution of Flexible load')
        disp('Please, Enter a new starting time')
        fprintf('Latest Starting Time is %d \n', (n-non_zero_idx))
    
        return
    else
        disp('Everything works fine')
    end
    
    %Restricts TSA execution until after start time 
    P1( 1 : start_time, :) = inf; %
    
    %restrict TSA execution till end of day
    P1( s+2: end,  : ) = inf;
    %P2( s1+2: end,  : ) = inf;
    %%% end of flexible load
       


    E = P1;

end 




