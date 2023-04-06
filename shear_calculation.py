import math


def main():
    # Define input data
    data = {
        'h': [1.3, 0.455],
        'bw': [1.5, 1.5],
        'fck': [90, 90],
        'fys': [500, 500],
        'fs': [1.15, 1.15],
        'fc': [1.5, 1.5],
        'ds': [16, 14],
        'ss': [100, 100],
        'Ast': [60.32, 15.39],
        'dtz': [14, 14],
        'stz': [200, 200],
        'Azt': [69.3, 38.5],
        'dty': [14, 14],
        'sty': [200, 200],
        'Ayt': [30.8, 7.7],
        'alpha_s': [33.69 * math.pi / 180, 45 * math.pi / 180],
        'b_span': [2.69, 1.40]
    }

    # Initialize output data
    Acro = [[0, 0] for _ in range(2)]
    peri = [[0, 0] for _ in range(2)]
    eff_wt = [[0, 0] for _ in range(2)]
    eff_wt_aux = [[0.001 * (35 + 16 + 40 + 16 / 2), 0.001 * (35 + 16 / 2)] for _ in range(2)]
    peritor = [[0, 0] for _ in range(2)]
    effz = [[0, 0] for _ in range(2)]
    levz = [[0, 0] for _ in range(2)]
    effy = [[0, 0] for _ in range(2)]
    levy = [[0, 0] for _ in range(2)]
    Aenc = [[0, 0] for _ in range(2)]
    cotant = [[0, 0] for _ in range(2)]
    maxctany = [[0, 0] for _ in range(2)]
    maxctanz = [[0, 0] for _ in range(2)]
    fywd = [[0, 0] for _ in range(2)]
    fyd = [[0, 0] for _ in range(2)]
    fcd = [[0, 0] for _ in range(2)]
    fcm = [[0, 0] for _ in range(2)]
    fctm = [[0, 0] for _ in range(2)]
    v1 = [[0, 0] for _ in range(2)]
    acw = [[0, 0] for _ in range(2)]
    v = [[0, 0] for _ in range(2)]
    v_oper = []

    for jj in range(2):
        # Acro
        Acro.append([data['h'][jj] * data['bw'][jj], 0])
        # peri
        peri.append([2 * (data['h'][jj] + data['bw'][jj]), 0])
        # eff_wt
        if Acro[jj][0] / peri[jj][0] > eff_wt_aux[jj][0]:
            eff_wt.append([Acro[jj][0] / peri[jj][0], 0])
        else:
            eff_wt.append([eff_wt_aux[jj][0], 0])
        # peritor
        peritor.append([2 * ((data['h'][jj] - 2 * eff_wt[jj][0]) + ((data['bw'][jj] - 2 * eff_wt[jj][0]))), 0])
        # effz
        effz.append([0.9 * data['h'][jj], 0])
        # levz
        levz.append([0.9 * effz[jj][0], 0])
        # effy
        effy.append([0.9 * data['bw'][jj], 0])
        # levy
        levy.append([0.9 * effy[jj][0], 0])
        # Aenc
        Aenc.append([(data['h'][jj] - eff_wt[jj][0]) * (data['bw'][jj] - eff_wt[jj][0]), 0])
        # cotant
        maxctany.append([data['b_span'][jj] / data['bw'][jj], 0])
        maxctanz.append([data['b_span'][jj] / data['h'][jj], 0])
        if (1 / math.tan(data['alpha_s'][jj])) < maxctany[jj][0] and (1 / math.tan(data['alpha_s'][jj])) < maxctanz[jj][0]:
            if (1 / math.tan(data['alpha_s'][jj])) > 1:
                cotant.append([1 / math.tan(data['alpha_s'][jj]), 0])
            else:
                cotant.append([1, 0])
        elif maxctany[jj][0] < maxctanz[jj][0]:
            if maxctany[jj][0] > 1:
                cotant.append([maxctany[jj][0], 0])
            else:
                cotant.append([1, 0])
        else:
            if maxctanz[jj][0] > 1:
                cotant.append([maxctanz[jj][0], 0])
            else:
                cotant.append([1, 0])
        # fywd

    # fywd
    v_oper.append(data['fys'][jj] / data['fs'][jj])
    fyd.append([data['fys'][jj] / data['fs'][jj], 0])
    fywd.append([min(fyd[jj][0], 0.8 * data['fys'][jj]), 0])
    # fcd
    fcd.append([data['fck'][jj] / data['fc'][jj], 0])
    # fcm
    fcm.append([data['fck'][jj] + 8, 0])
    # fctm
    if data['fck'][jj] <= 50:
        fctm.append([0.3 * data['fck'][jj], 0])
    else:
        fctm.append([2.12 * math.log(1 + fcm[jj][0] / 10), 0])
    # v1
    if 0.9 - data['fck'][jj] / 200 > 0.5:
        v1.append([0.9 - data['fck'][jj] / 200, 0])
    else:
        v1.append([0.5, 0])
    # alphacw
    acw.append([1, 0])
    # v
    v.append([0.6 * (1 - data['fck'][jj] / 250), 0])
