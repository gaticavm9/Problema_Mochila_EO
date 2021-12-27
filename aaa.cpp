solution_t iniSolution(int nItems){
  solution_t sol;

  for(int i=0; i<nItems; i++)
    if (fRandetol( < 0.5)
   |sol.x.push_back (0);
    else
      sol.x.push_back (1);
  return sol;


solution_t evalSolution(data_t d, solution_t s, int iter) {
  s.iteration=iter;
  s.z=0;
  s.c=0;
  for(int i=0; i<d.n;i++) {
    s.z=s.z+(d.p[i]*s.x[i]);
    s.c=s.c+(d.w[i]*s.x[i]);
 if (s.c <= d.c)
    s. feasible=true;
 else
    s.feasible=false;
  return s;
}


solution_t iniBestSolution(int nItems){
  solution_t sol;
  sol.iteration=0;
  sol.feasible=false;
  sol.z=0;
  sol.c=INT_MAX;
  for(int i=0;i<nItems; i++)
    sol.x.push_back(0);
  return sol;
}



void iniProbabilityVector(int n, float t) {
  for(int i=0; i<n; i++)
   p. push_back(pow(i+1,-t));
}

fitness_t calculateFitness(data_t d) {
  fitness_t f;
  for (int i=0; i < d.n; i++)
    f.push_back(make_pair((double)d.p[i]/(double)d.w[i],i));
  return f;
}                                     



roulette_t createRoulette(int size) {
  roulette_t ruleta;
 double sum=0.0;
 vector <double> paso;
  ruleta.r.clear();
  ruleta.size=size;
 for(int i=0; i<ruleta.size; i++)
    sum=sum+p[i];
 for(int i=0; i<ruleta.size; i++)
    paso.push_back(p[i]/sum);
  ruleta.r.push_back(paso[0]);
 for(int i=1; i<ruleta.size; i++) 
    ruleta.r.push_back(ruleta.r[i-1]+paso[i]);
  return ruleta;
}



int rouletteDraw(roulette t ruleta){
  int i=0;
  bool leave=false;
  double draw=0.0;

  draw=fRandotol();
  //cout « "\ndraw: " << draw < endl;
  do {
    if(draw <= ruleta.r[i]){
        leave=true;
      } else {
        i++;
  } while (leave==false);
  return i;
}


fitness_t feasibleFitness (vector<int> x, fitness_t f) {
  fitness_t sorted_f;
  for(unsigned int i=0; i<x.size(); i++)
    if(x[i]==0)
      sorted f.push_back(make_pair(f[i].first, f[i].second));
     sort(sorted f.begin(), sorted_f.end(), comparisonTopDown);
  return sorted_f;
}
                                        

fitness_t infeasibleFitness (vector<int> x, fitness_t f) {
  fitness_t sorted_f;
  for(unsigned int i-0; i<x.size(); i++)
    if(x[i]==1)
      sorted f.push_back(make_pair(f[i].first, f[i].second));
    sort(sorted f.begin(), sorted_f.end(), comparisonBottomup);
  return sorted f;