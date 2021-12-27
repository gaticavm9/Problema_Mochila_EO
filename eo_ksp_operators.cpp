








fitness_t feasibleFitness (vector<int> x, fitness_t f) {
  fitness t sorted_f;
 for (unsigned int i=0; i<x.size(0; i++)
    if(x[i]==0)
       sorted f.push_back(make_pair(f[i].first,f[i].second));
    sort(sorted_f.begin(), sorted_f.end (), comparisonTopDown);
  return sorted_f;
}


fitness_t infeasibleFitness (vector<int> x, fitness_t f) {
  fitness t sorted_f;
  for(unsigned int i=8; i<x.size(0; i++)
    if(x[i]==1)
      sorted f.push_back (make_pair(f[i].first,f[i].second));
    sort(sorted_f.begin(), sorted f.end(), comparisonBottomUp);
  return sorted f;