# 评测结果

`results.json`使用官方附件结构：schema_version=1、team_id="1385264"；results下clean/randomized各50任务，每任务仅含attempts与successes。attempts为实际完成的有效policy回合，不是计划数或专家采集次数。

|场景|指令|计划|attempts|successes|有效回合成功率|专家阻塞|初始化异常|
|---|---|---:|---:|---:|---:|---:|---:|
|clean|seen|500|455|65|14.29%|45|0|
|randomized|unseen|500|423|43|10.17%|76|1|

按任务的实际次数均可从evidence/quick-results-with-rounds.json、per_task.csv及raw/两组逐回合日志核对。原自定义JSON移至evidence保留，未改写历史回合或异常。专家阻塞不属于已执行policy，初始化异常也未记为有效回合。最新官方Q&A要求专家失败不应阻止policy；历史差异保留披露，没有因更换结果格式而得到修正。

unseen-original-export是原始证据存档；原始说明中的历史准备文字应结合冻结配置与实际事件解读。最终randomized+unseen确由真实重测获得，旧randomized+seen不计入本次结果。来源映射见aggregation-manifest.json。

受算力限制，每任务每场景仅计划10次。官方Q&A允许不足100次时按实测提交、最终以官方复测为准；本文件不声明官方排名分数。
