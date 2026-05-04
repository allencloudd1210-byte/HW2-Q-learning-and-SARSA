# HW2: Q-learning and SARSA on Cliff Walking

這個 repository 是強化學習作業 `HW2: Q-learning 與 SARSA 演算法之比較研究` 的實作與實驗結果。

## 專案內容

本作業在經典 `Cliff Walking` 環境中，比較兩種方法：

- `Q-learning`：Off-policy
- `SARSA`：On-policy

比較重點包含：

- 每回合累積獎勵（Total Reward）
- 收斂速度
- 最終學習路徑
- 策略是否偏向冒險或保守
- 學習過程的穩定性

## 環境設定

- Grid size: `4 x 12`
- Start: 左下角
- Goal: 右下角
- Cliff: 起點與終點之間底部區域
- Step reward: `-1`
- Cliff reward: `-100` 並回到起點

主實驗參數：

- `episodes = 500`
- `epsilon = 0.1`
- `alpha = 0.1`
- `gamma = 0.9`
- `runs = 30`

## 檔案結構

- `hw2_cliff_walking.py`: Cliff Walking、Q-learning、SARSA、訓練流程與繪圖程式
- `hw2_report.md`: 完整作業報告
- `outputs/reward_curves.png`: 累積獎勵曲線
- `outputs/final_paths.png`: 最終路徑比較圖
- `outputs/summary.json`: 主實驗數值摘要
- `conversation_log.txt`: 本次對話與工作記錄

## 執行方式

```bash
python3 hw2_cliff_walking.py
```

若只想快速測試：

```bash
python3 hw2_cliff_walking.py --runs 5
```

## 實驗結果摘要

主實驗結果顯示：

- `Q-learning` 收斂代理指標較快
- `SARSA` 較穩定，reward 波動較小
- `Q-learning` 學到較短、較貼近懸崖的路徑
- `SARSA` 學到較安全、較保守的路徑

根據 `outputs/summary.json` 的主實驗結果：

| 指標 | Q-learning | SARSA |
| --- | ---: | ---: |
| 最後 100 回合平均 reward | `-48.54` | `-24.32` |
| 最後 100 回合 reward 標準差 | `12.15` | `4.27` |
| 最後 100 回合平均掉崖次數 | `0.317` | `0.050` |
| 代表性最終路徑長度 | `13` | `15` |

## 視覺化結果

### Reward Curves

![Reward Curves](outputs/reward_curves.png)

### Final Paths

![Final Paths](outputs/final_paths.png)

## 結論

在 `Cliff Walking` 這種高風險環境下：

- `Q-learning` 傾向學到理論上更短、更有效率，但風險較高的策略
- `SARSA` 傾向學到更安全、更穩定的策略

如果更重視訓練過程的穩定性與安全性，`SARSA` 較合適；如果更重視最終路徑效率，`Q-learning` 仍具有優勢。
