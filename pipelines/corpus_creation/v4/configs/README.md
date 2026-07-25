# v4実行設定

`config.json`と`config.example.json`はv4直下の標準設定として残し、このフォルダーには用途限定の実行設定を置く。

- `pilot10.openai.json`：OpenAI生徒を使う少数件確認と、現在の120候補監査・Repairの再現設定
- `slice-a.openai.json`〜`slice-d.openai.json`：120候補を30件ずつ生成した分割ジョブ設定

設定内の相対パスはこのフォルダーを基準に解決されるため、問題・対応表・出力先には`../`を付けている。既存runを再開するときは対応する設定を使い、別条件では出力先を分ける。
