# TODO

- [ ] Improve train test splits - currently random, but doesn't show if we can extrapolate
    - [ ] 0.25 vertical split or a 0.25 horizontal split
    - [ ] 0.125 vertical + 0.125 horizontal split
    - [ ] [bottom/top]-[left/right] corner of the domain

- [ ] Separate the features into: `rbf(temperature[all])`, `rbf(precip[all])`, `rq(remaining)`. We don't expect this to improve results, but will allow us to assess the length scales etc.  

**What I've learnt**: Taking the product of RQ(core) / RQ(temp_precip) is important, otherwise low scores. Best combination is: 
$$\textbf{Index}\cdot(\textbf{Poly}[temp,alt] + \textbf{RQ}[core] \cdot \textbf{RQ}[T,P])$$ 
where RQ(precip/temp) does not have independent length scales (ARD). Using more combinations doesn't improve scores (both in test and mapped), but not splitting these and keeping everything under the same kernel ($RQ[core,T,P]$) results in poor estimates when compared to mapped results. 
Why could this be? We're forcing lengthscales of T,P to be the same. This might actually improve results since these are coarse data. While *core* contains variables that differen on different lengthscales. 