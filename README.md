# Research Network Transfer Performance Predictor (netperf-predict)

This respository containts two sets of analysis routines for predicting the percentage of retransmitted packets on network 
flows.  One directory contains code that applies random forest regression in order to predict the number of retransmitted 
packets on each flow, operating on timeseries data from the [tstat][tstat] tool, which outputs flow-like data.  The second 
directory also applies a random forest regression and also incorporates a "smoothing" routine that increases accuracy in some 
situations.

This code was developed at Lawrence Berkeley National Laboratory as part of the NSF IRNC-funded ["NetSage" project][netsage], 
award number [OAC-1540933][award] (Lead PI at Indiana University, Jennifer Schopf; Co-PI and Berkeley Lab Lead, Sean Peisert).

Results of using this code are described in the following paper:

```
Anna Giannakou, Daniel Gunter, and Sean Peisert, "Flowzilla: A Methodology for Detecting Data Transfer Anomalies in 
Research Networks," Proceedings of the 5th Innovate the Network for Data-Intensive Science (INDIS) Workshop, Dallas, TX, 
November 11, 2018.
```

[Download PDF][flowzilla]

Contributors to this code repository at the Berkeley Lab included:

* [Anna Giannakou](https://crd.lbl.gov/anna-giannakou/) (Developer / Postdoc)
* [Dipankar Dwivedi](https://eesa.lbl.gov/profiles/dipankar-dwivedi/) (Developer / Research Scientist)
* [Sean Peisert](http://crd.lbl.gov/sean-peisert/) (Project Co-PI and Berkeley Lab Lead)

Questions about the project to the code's contributors are welcome.

[flowzilla]: http://www.cs.ucdavis.edu/~peisert/research/2018-INDIS-Flowzilla
[indis]: https://scinet.supercomputing.org/workshop/sc18-program
[tstat]: http://tstat.polito.it
[netsage]: http://www.netsage.global
[award]: https://www.nsf.gov/awardsearch/showAward?AWD_ID=1540933
