#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data from Excel file
FRU = pd.read_excel(r"C:\Users\madhu\Documents\NIRF University Ranking Calculation.xlsx", sheet_name=0)
PU_QP = pd.read_excel(r"C:\Users\madhu\Documents\NIRF University Ranking Calculation.xlsx", sheet_name=1)
#IPR = pd.read_excel(r"C:\Users\madhu\Documents\NIRF University Ranking Calculation.xlsx", sheet_name=2)
FPPP = pd.read_excel(r"C:\Users\madhu\Documents\NIRF University Ranking Calculation.xlsx", sheet_name=3)
GPHD = pd.read_excel(r"C:\Users\madhu\Documents\NIRF University Ranking Calculation.xlsx", sheet_name=4)
PCS = pd.read_excel(r"C:\Users\madhu\Documents\NIRF University Ranking Calculation.xlsx", sheet_name=5)

# Split the data into input and output
y = FRU[['FRU']]
x = FRU[['Capex_avg', 'Opex_avg']]
y11 = PU_QP[['PU']]
x11 = PU_QP[['Publications', 'faculty_2018']]
y12 = PU_QP[['QP']]
x12 = PU_QP[['Publications', 'Citations', 'Top25','faculty_2018']]
#y2 = IPR[['IPR']]
#x2 = IPR[['Patent_Granted', 'Patent_Published']]
y3 = FPPP[['FPPP']]
x3 = FPPP[['Research', 'Consultancy', 'Executive']]
y4 = GPHD[['GPHD']]
x4 = GPHD[['FT_grad']]
y5 = PCS[['PCS']]
x5 = PCS[['A', 'B', 'C']]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x11_train, x11_test, y11_train, y11_test = train_test_split(x11,y11, test_size=0.2,random_state=31)
x12_train, x12_test, y12_train, y12_test = train_test_split(x12,y12, test_size=0.2,random_state=31)
#x2_train, x2_test, y2_train, y2_test=train_test_split(x2,y2, test_size=0.2,random_state=31)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3, test_size=0.2,random_state=31)
x4_train, x4_test, y4_train, y4_test=train_test_split(x4,y4, test_size=0.2,random_state=31)
x5_train, x5_test, y5_train, y5_test=train_test_split(x5,y5, test_size=0.2,random_state=31)

# Train the random forest regression model for prediction
fru_model = RandomForestRegressor(n_estimators=1000, random_state=31)
fru_rf = fru_model.fit(x_train, y_train.values.ravel())
pu_model = RandomForestRegressor(n_estimators=1000, random_state=31)
pu_rf = pu_model.fit(x11_train, y11_train.values.ravel())
qp_model = RandomForestRegressor(n_estimators=1000, random_state=31)
qp_rf = qp_model.fit(x12_train, y12_train.values.ravel())
#ipr_model = RandomForestRegressor(n_estimators=1000, random_state=31)
#ipr_rf = ipr_model.fit(x2_train, y2_train.values.ravel())
fppp_model = RandomForestRegressor(n_estimators=1000, random_state=31)
fppp_rf = fppp_model.fit(x3_train, y3_train.values.ravel())
gphd_model = RandomForestRegressor(n_estimators=1000, random_state=31)
gphd_rf = gphd_model.fit(x4_train, y4_train.values.ravel())
pcs_model = RandomForestRegressor(n_estimators=1000, random_state=31)
pcs_rf = pcs_model.fit(x5_train, y5_train.values.ravel())


def predict_SS(SI, TE, ft):
    students = max(SI,TE)
    Ratio = min(TE/SI,1)
    if students >=20000:
        criteria = 15
    elif students <=19999 and students >=10000:
        criteria = 13.5
    elif students <=9999 and students >=5000:
        criteria = 12
    elif students <=4999 and students >=4000:
        criteria = 10.5
    elif students <=3999 and students >=3000:
        criteria = 9
    elif students <=2999 and students >=2000:
        criteria = 7.5
    elif students <=1999 and students >=1000:
        criteria = 6
    student_score = round(Ratio*criteria,2)
    if ft>=2000:
        PhD_score = 5
    elif ft<=1999 and ft>=1000:
        PhD_score = 4
    elif ft<=999 and ft>=500:
        PhD_score = 3
    elif ft<=499 and ft>=250:
        PhD_score = 2
    elif ft<=249 and ft>=50:
        PhD_score = 1
    elif ft<=49 and ft>=25:
        PhD_score = 0.5
    else:
        PhD_score = 0
    SS = student_score + PhD_score
    return SS

def predict_FSR(Nfaculty, SI, ft):
    N = SI + ft
    ratio = Nfaculty / N
    FSR = min((round(25 * (15 * ratio),2)),25)
    #if FSR > 25:
        #FSR = 25
    #else:
        #FSR = FSR
    if ratio < 0.02:
        FSR = 0
    return FSR

def predict_FQE(SI, Nfaculty, phd, ft_exp1, ft_exp2, ft_exp3):
    calc_FSR=SI/15
    Faculty = max(calc_FSR,Nfaculty)
    FRA=(phd/Faculty)
    if FRA<0.95:
        FQ=10*(FRA/0.95)
    else:
        FQ=10
    F1=ft_exp1/Faculty
    F2=ft_exp2/Faculty
    F3=ft_exp3/Faculty
    FE_cal=(3*min((3*F1),1))+(3*min((3*F2),1))+(4*min((3*F3),1))
    if F1==F2==F3:
        FE=10
    else:
        FE = FE_cal
    FQE=round(FQ+FE,2)
    return FQE

def predict_FRU(Capex_avg, Opex_avg):
    FRU = round(fru_model.predict([[Capex_avg, Opex_avg]])[0],2)
    return FRU

def predict_PU(Publications, Nfaculty):
    # Make a prediction using the trained model for SS
    PU = round(pu_model.predict([[Publications, Nfaculty]])[0],2)
    return PU

def predict_QP(Publications,Citations,Top25,Nfaculty):
    # Make a prediction using the trained model for SS
    QP = round(qp_model.predict([[Publications,Citations,Top25,Nfaculty]])[0],2)
    return QP

def predict_IPR(PG,PP):
    if PG >=75:
        PG_score = 10
    elif PG <=74 and PG >=50:
        PG_score = 8
    elif PG <=49 and PG >=25:
        PG_score = 6
    elif PG <=24 and PG >=10:
        PG_score = 4
    elif PG <=9 and PG >=5:
        PG_score = 2
    elif PG <=4 and PG >=1:
        PG_score = 1
    elif PG == 0:
        PG_score = 0
    
    if PP >=300:
        PP_score = 5
    elif PP <=299 and PP >=200:
        PP_score = 4
    elif PP <=199 and PP >=150:
        PP_score = 3
    elif PP <=149 and PP >=50:
        PP_score = 2
    elif PP <=49 and PP >=10:
        PP_score = 1
    elif PP <=9 and PP >=1:
        PP_score = 0.5
    elif PP == 0:
        PP_score = 0
    
    IPR = PG_score + PP_score
    return IPR


def predict_FPPP(Research,Consultancy,Executive):
    FPPP = round(fppp_model.predict([[Research,Consultancy,Executive]])[0],2)
    return FPPP

def predict_GUE(graduated1,graduated2,graduated3,si1,si2,si3):
    year1 = graduated1/si1
    year2 = graduated2/si2
    year3 = graduated3/si3
    avg=(year1+year2+year3)/3
    a=avg/0.8
    GUE=round((min(a,1)*60),2)
    return GUE

def predict_GPHD(FT_grad):
    # Make a prediction using the trained model for SS
    GPHD = round(gphd_model.predict([[FT_grad]])[0],2)
    return GPHD

def predict_RD(SI,TE,other_state,other_country):
    students = max(SI,TE)
    state = (other_state/students)*25
    country = (other_country/students)*5
    RD = round((state + country),2)
    return RD

def predict_WD(WS,WF,SI,TE,Nfaculty):
    calc_FSR=(max(SI,TE))/15
    Faculty = max(calc_FSR,Nfaculty)
    student_ratio=WS/SI
    faculty_ratio=WF/Faculty
    a1=min(((student_ratio)/0.5),1)
    b1=min(((faculty_ratio)/0.2),1)
    WD=round(((15*a1)+(15*b1)),2)
    return WD

def predict_ESCS(SI, socio_economic, reimbursed):
    Reimbursed_ratio = (reimbursed/socio_economic)
    Student_ratio = (socio_economic/SI)
    ESCS = round((Reimbursed_ratio*Student_ratio*10),2)
    #ESCS = tuple(round(val, 2) for val in (Reimbursed_ratio, Student_ratio*10))
    return ESCS

def predict_PCS(A,B,C):
    PCS = round(pcs_model.predict([[A,B,C]])[0],2)
    return PCS

def predict_all(ug31,ug32,ug33,ug41,ug42,ug43,ug44,ug51,ug52,ug53,ug54,ug55,ug61,ug62,ug63,ug64,ug65,ug66,pg11,pg21,pg22,pg31,pg32,pg33,pg51,pg52,pg53,pg54,pg55,pg61,pg62,pg63,pg64,pg65,pg66,UG3_TE,UG4_TE,UG5_TE,UG6_TE,PG1_TE,PG2_TE,PG3_TE,PG5_TE,PG6_TE,ft, Nfaculty, phd,ft_exp1,ft_exp2,ft_exp3, L1,L2,L3,Lab1,Lab2,Lab3,W1,W2,W3,Studio1,Studio2,Studio3,O1,O2,O3, S1,S2,S3,I1,I2,I3,Seminar1,Seminar2,Seminar3, OE, faculty_2018,Publications,Citations,Top25, PG,PP, RF1,RF2,RF3,CF1,CF2,CF3,Executive1,Executive2,Executive3, graduated_ug31,graduated_ug32,graduated_ug33,graduated_ug41,graduated_ug42,graduated_ug43,graduated_ug51,graduated_ug52,graduated_ug53,graduated_ug61,graduated_ug62,graduated_ug63,graduated_pg11,graduated_pg12,graduated_pg13,graduated_pg21,graduated_pg22,graduated_pg23,graduated_pg31,graduated_pg32,graduated_pg33,graduated_pg51,graduated_pg52,graduated_pg53,graduated_pg61,graduated_pg62,graduated_pg63,si_ug31,si_ug32,si_ug33,si_ug41,si_ug42,si_ug43,si_ug51,si_ug52,si_ug53,si_ug61,si_ug62,si_ug63,si_pg11,si_pg12,si_pg13,si_pg21,si_pg22,si_pg23,si_pg31,si_pg32,si_pg33,si_pg51,si_pg52,si_pg53,si_pg61,si_pg62,si_pg63, FT_grad1,FT_grad2,FT_grad3, state1,state2,state3,state4,state5,state6,state7,state8,state9,country1,country2,country3,country4,country5,country6,country7,country8,country9, WS,WF, Socially_challenged1,Socially_challenged2,Socially_challenged3,Socially_challenged4,Socially_challenged5,Socially_challenged6,Socially_challenged7,Socially_challenged8,Socially_challenged9,economically_challenged1,economically_challenged2,economically_challenged3,economically_challenged4,economically_challenged5,economically_challenged6,economically_challenged7,economically_challenged8,economically_challenged9,reimbursed1,reimbursed2,reimbursed3,reimbursed4,reimbursed5,reimbursed6,reimbursed7,reimbursed8,reimbursed9, A,B,C, PR):
    SI = ug31+ug32+ug33+ug41+ug42+ug43+ug44+ug51+ug52+ug53+ug54+ug55+ug61+ug62+ug63+ug64+ug65+ug66+pg11+pg21+pg22+pg31+pg32+pg33+pg51+pg52+pg53+pg54+pg55+pg61+pg62+pg63+pg64+pg65+pg66
    TE = UG3_TE+UG4_TE+UG5_TE+UG6_TE+PG1_TE+PG2_TE+PG3_TE+PG5_TE+PG6_TE
    ss = predict_SS(SI, TE, ft)
    fsr = predict_FSR(Nfaculty, SI, ft)
    fqe = predict_FQE(SI, Nfaculty, phd, ft_exp1, ft_exp2, ft_exp3)
    cap_year1 = (L1+Lab1+W1+Studio1+O1)/SI
    cap_year2 = (L2+Lab2+W2+Studio2+O2)/SI
    cap_year3 = (L3+Lab3+W3+Studio3+O3)/SI
    Capex_avg = ((cap_year1+cap_year2+cap_year3)/3)
    opr_year1 = (S1+I1+Seminar1)/SI
    opr_year2 = (S2+I2+Seminar2)/SI
    opr_year3 = (S3+I3+Seminar3)/SI
    Opex_avg = ((opr_year1 + opr_year2 + opr_year3)/3)
    fru = predict_FRU(Capex_avg, Opex_avg)
    oe = OE
    tlr = round((ss + fsr + fqe + fru + OE),2)
    pu = predict_PU(Publications, faculty_2018)
    qp = predict_QP(Publications,Citations,Top25,faculty_2018)
    ipr = predict_IPR(PG,PP)
    Research = (RF1+RF2+RF3)/3
    Consultancy = (CF1+CF2+CF3)/3
    Executive = (Executive1+Executive2+Executive3)/3
    fppp = predict_FPPP(Research,Consultancy,Executive)
    rp = round((pu + qp + ipr + fppp),2)
    graduated1 = graduated_ug31+graduated_ug41+graduated_ug51+graduated_ug61+graduated_pg11+graduated_pg21+graduated_pg31+graduated_pg51+graduated_pg61
    graduated2 = graduated_ug32+graduated_ug42+graduated_ug52+graduated_ug62+graduated_pg12+graduated_pg22+graduated_pg32+graduated_pg52+graduated_pg62
    graduated3 = graduated_ug33+graduated_ug43+graduated_ug53+graduated_ug63+graduated_pg13+graduated_pg23+graduated_pg33+graduated_pg53+graduated_pg63
    si1 = si_ug31+si_ug41+si_ug51+si_ug61+si_pg11+si_pg21+si_pg31+si_pg51+si_pg61
    si2 = si_ug32+si_ug42+si_ug52+si_ug62+si_pg12+si_pg22+si_pg32+si_pg52+si_pg62
    si3 = si_ug33+si_ug43+si_ug53+si_ug63+si_pg13+si_pg23+si_pg33+si_pg53+si_pg63
    gue = predict_GUE(graduated1,graduated2,graduated3,si1,si2,si3)
    FT_grad = (FT_grad1+ FT_grad2+ FT_grad3)/3
    gphd = predict_GPHD(FT_grad)
    go = round((gue + gphd),2)
    other_state = state1+state2+state3+state4+state5+state6+state7+state8+state9
    other_country = country1+country2+country3+country4+country5+country6+country7+country8+country9
    rd = predict_RD(SI,TE,other_state,other_country)
    wd = predict_WD(WS,WF,SI,TE,Nfaculty)
    socio_economic = Socially_challenged1+Socially_challenged2+Socially_challenged3+Socially_challenged4+Socially_challenged5+Socially_challenged6+Socially_challenged7+Socially_challenged8+Socially_challenged9+economically_challenged1+economically_challenged2+economically_challenged3+economically_challenged4+economically_challenged5+economically_challenged6+economically_challenged7+economically_challenged8+economically_challenged9
    reimbursed = reimbursed1+reimbursed2+reimbursed3+reimbursed4+reimbursed5+reimbursed6+reimbursed7+reimbursed8+reimbursed9
    escs = predict_ESCS(SI, socio_economic, reimbursed)
    pcs = predict_PCS(A,B,C)
    oi = round((rd + wd + escs + pcs),2)
    pr = PR
    Overall_score = round(((tlr * 0.3) + (rp * 0.3) + (go * 0.2) + (oi * 0.1) + (pr * 0.1)),2)
    
    return(ss,fsr,fqe,fru,oe,tlr,pu,qp,ipr,fppp,rp,gue,gphd,go,rd,wd,escs,pcs,oi,pr,Overall_score)

# Create a Gradio interface
inputs = [
    gr.inputs.Number(label="Sanctioned Intake UG 3 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 3 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 3 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 4 year-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 5 year-year5", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 6 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 6 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 6 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 6 year-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 6 year-year5", default=60),
    gr.inputs.Number(label="Sanctioned Intake UG 6 year-year6", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 1 year", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 2 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 2 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 3 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 3 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 3 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG integrated-year5", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year1", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year2", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year3", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year4", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year5", default=60),
    gr.inputs.Number(label="Sanctioned Intake PG 6 year-year6", default=60),
    gr.inputs.Number(label="Total Enrollment - UG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment - UG 4 year", default=60),
    gr.inputs.Number(label="Total Enrollment - UG 5 year", default=60),
    gr.inputs.Number(label="Total Enrollment - UG 6 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG 1 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG 2 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment - PG integrated", default=60),
    gr.inputs.Number(label="Total Enrollment - PG 6 year", default=60),
    gr.inputs.Number(label="Number of PhD Enrolled Full Time", default=10),
    gr.inputs.Number(label="No. of Full Time Regular Faculty", default=20),
    gr.inputs.Number(label="No. of faculty with PhD", default=5),
    gr.inputs.Number(label="No. of full time regular faculty with Experience up to 8 years", default=5),
    gr.inputs.Number(label="No. of full time regular faculty with Experience between 8+ to 15 years", default=10),
    gr.inputs.Number(label="No. of full time regular faculty with Experience > 15 years", default=15),
    gr.inputs.Number(label="Annual Expenditure on Library-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Library-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Library-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Laboratory-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Laboratory-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Laboratory-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Workshop-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Workshop-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Workshop-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Studio-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Studio-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Studio-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Others-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Others-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Others-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Salary-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Salary-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Salary-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Infrastructure-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Infrastructure-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Infrastructure-year3", default=60),
    gr.inputs.Number(label="Annual Expenditure on Seminar-year1", default=60),
    gr.inputs.Number(label="Annual Expenditure on Seminar-year2", default=60),
    gr.inputs.Number(label="Annual Expenditure on Seminar-year3", default=60),
    gr.inputs.Number(label="OE", default=60),
    gr.inputs.Number(label="No. of Full Time Regular Faculty", default=20),
    gr.inputs.Number(label="No. of Publications", default=60),
    gr.inputs.Number(label="No. of citations", default=60),
    gr.inputs.Number(label="No. of Top25 percentage", default=60),
    gr.inputs.Number(label="No. of Patent_Granted", default=60),
    gr.inputs.Number(label="No. of Patent_Published", default=60),
    gr.components.Number(label="Amount received in sponsored research - year1", value=29557741),
    gr.components.Number(label="Amount received in sponsored research - year2", value=55891434),
    gr.components.Number(label="Amount received in sponsored research - year3", value=112200598),
    gr.components.Number(label="Amount received in consultancy projects - year1", value=25325000),
    gr.components.Number(label="Amount received in consultancy projects - year2", value=116740000),
    gr.components.Number(label="Amount received in consultancy projects - year3", value=189056000),
    gr.components.Number(label="Amount earned in EDP - year1", value=584100000),
    gr.components.Number(label="Amount earned in EDP - year2", value=597984756),
    gr.components.Number(label="Amount earned in EDP - year3", value=419501486),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 3 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 3 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 3 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 4 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 4 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 4 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 5 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 5 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 5 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 6 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 6 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(UG 6 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 1 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 1 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 1 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 2 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 3 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 3 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 3 year) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG integrated) -  year3", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 6 year) -  year1", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 6 year) -  year2", default=60),
    gr.inputs.Number(label="No. of students graduated in minimum stipulated time(PG 6 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 3 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 3 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 3 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 4 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 4 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 4 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 5 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 5 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 5 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 6 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 6 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(UG 6 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 1 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 1 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 1 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 2 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 3 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 3 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 3 year) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG integrated) -  year3", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 6 year) -  year1", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 6 year) -  year2", default=60),
    gr.inputs.Number(label="No. of sanctioned intake(from placement & HS table)(PG 6 year) -  year3", default=60),
    gr.inputs.Number(label="No. of PhD. students graduated - year1", default=60),
    gr.inputs.Number(label="No. of PhD. students graduated - year2", default=60),
    gr.inputs.Number(label="No. of PhD. students graduated - year3", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - UG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - UG 4 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - UG 5 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - UG 6 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG 1 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG 2 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG integrated", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other states - PG 6 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - UG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - UG 4 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - UG 5 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - UG 6 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG 1 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG 2 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG 3 year", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG integrated", default=60),
    gr.inputs.Number(label="Total Enrollment of students from other countries - PG 6 year", default=60),
    gr.inputs.Number(label="No. of women students", default=60),
    gr.inputs.Number(label="No. of women faculty", default=60),
    gr.inputs.Number(label="No. of students socially challenged - UG 3 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - UG 6 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG 1 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG integrated", default=60),
    gr.inputs.Number(label="No. of students socially challenged - PG 6 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - UG 3 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - UG 6 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG 1 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG integrated", default=60),
    gr.inputs.Number(label="No. of students economically challenged - PG 6 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - UG 3 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - UG 4 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - UG 5 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - UG 6 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - PG 1 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - PG 2 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - PG 3 year", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - PG integrated", default=60),
    gr.inputs.Number(label="No. of students being provided full tuition fee reimbursement from government, institutions and private bodies - PG 6 year", default=60),
    gr.components.Number(label="Lifts/Ramps", value=80),
    gr.inputs.Dropdown(choices = ["1", "0"],label="Walking aids", default=1),
    gr.components.Number(label="Specially designed toilets for handicapped students", value=80),
    gr.components.Number(label="PR", value=94.74)
]
output1 = gr.outputs.Textbox(label="SS")
output2 = gr.outputs.Textbox(label="FSR")
output3 = gr.outputs.Textbox(label="FQE")
output4 = gr.outputs.Textbox(label="FRU")
output5 = gr.outputs.Textbox(label="OE")
output_tlr = gr.outputs.Textbox(label="TLR")
output6 = gr.outputs.Textbox(label="PU")
output7 = gr.outputs.Textbox(label="QP")
output8 = gr.outputs.Textbox(label="IPR")
output9 = gr.outputs.Textbox(label="FPPP")
output_rp = gr.outputs.Textbox(label="RP")
output10 = gr.outputs.Textbox(label="GUE")
output11 = gr.outputs.Textbox(label="GPHD")
output_go = gr.outputs.Textbox(label="GO")
output12 = gr.outputs.Textbox(label="RD")
output13 = gr.outputs.Textbox(label="WD")
output14 = gr.outputs.Textbox(label="ESCS")
output15 = gr.outputs.Textbox(label="PCS")
output_oi = gr.outputs.Textbox(label="OI")
output_PR = gr.outputs.Textbox(label="PR")
output_score = gr.outputs.Textbox(label="Overall_score")

gradio_interface = gr.Interface(fn=predict_all, inputs=inputs, outputs=[output1, output2, output3, output4, output5, output_tlr, output6, output7, output8, output9, output_rp, output10, output11, output_go, output12, output13, output14, output15, output_oi, output_PR, output_score], title="University NIRF Score Calculation", 
                                description="Enter the input parameters to predict Overall score")

#url = deploy(gradio_interface, share=True)
gradio_interface.launch(share=True)
 


# In[ ]:




