#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:33:18 2017

@author: shugo
"""


import numpy as np
from Scripts.interface import TrajecSimu_UI
from concurrent.futures import ProcessPoolExecutor

# 並列に実行される関数
def run_loop(args):
    n_winddirec, max_windspeed, windspeed_step = args
    config_filename = 'Parameters_csv/23no.csv'
    mysim = TrajecSimu_UI(config_filename, 'izu')
    mysim.run_loop(n_winddirec, max_windspeed, windspeed_step)

if __name__ == '__main__':
    # raa csvファイルのパスとファイル名を定義
    config_filename = 'Parameters_csv/23no.csv'

    # run_loop関数の引数のリストを作成
    args_list = [(i, 8, 1) for i in range(8)]  # 必要に応じてrangeの範囲を調整してください

    # 最大プロセス数を指定
    max_processes = 4  # 例として4つのプロセスを同時に実行

    # プロセスを格納するリスト
    processes = []

    # プロセスプールを作成
    with ProcessPoolExecutor(max_processes) as executor:
        # 並列に実行
        futures = [executor.submit(run_loop, args) for args in args_list]

        # 完了したプロセスを確認し、プロセスを削除して新しいプロセスを開始
        for future in futures:
            if not future.running():
                processes.remove(future)
                if len(processes) < max_processes:
                    new_args = args_list[len(processes)]
                    new_process = executor.submit(run_loop, new_args)
                    processes.append(new_process)

    # すべてのプロセスの処理が終了するのを待つ
    for process in processes:
        process.result()
