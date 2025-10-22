$body = @{
    text1 = "A man is playing a guitar."
    text2 = "A person is strumming a musical instrument."
}

Invoke-RestMethod -Uri http://127.0.0.1:5000/calculate-similarity -Method POST -ContentType "application/json" -Body ($body | ConvertTo-Json)