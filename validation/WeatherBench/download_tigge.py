#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ti",
    "dataset": "tigge",
    "date": "2012-10-20/to/2012-11-15",
    "expver": "prod",
    "grid": "0.5/0.5",
    "levtype": "sfc",
    "origin": "ecmf/kwbc",
    "param": "165/166/228228",
    "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336",
    "time": "00:00:00",
    "type": "cf",
    "target": "output",
})
