library(mlbench)

save_to_csv = function(df,name){
  if (file.exists(name)) 
    #Delete file if it exists
    file.remove(name)
  
  write.csv(df, name)
}

load_datasets = function(){
  
  data("BostonHousing")
  data("BreastCancer")
  data("DNA")
  data("Glass")
  data("HouseVotes84")
  data("Ionosphere")
  data("Ozone")
  data("Satellite")
  data("Servo")
  data("Shuttle")
  data("Sonar")
  data("Soybean")
  data("Vehicle")
  data("Vowel")
  data("Zoo")
  save_to_csv(BostonHousing, 'data/bh.csv')
  save_to_csv(BreastCancer, 'data/bc.csv')
  save_to_csv(DNA, 'data/dna.csv')
  save_to_csv(Glass, 'data/gl.csv')
  save_to_csv(HouseVotes84, 'data/hv.csv')
  save_to_csv(Ionosphere, 'data/is.csv')
  save_to_csv(Ozone, 'data/on.csv')
  save_to_csv(Satellite, 'data/sl.csv')
  save_to_csv(Servo, 'data/sr.csv')
  save_to_csv(Shuttle, 'data/st.csv')
  save_to_csv(Sonar, 'data/sn.csv')
  save_to_csv(Soybean, 'data/sb.csv')
  save_to_csv(Vehicle, 'data/vc.csv')
  save_to_csv(Vowel, 'data/vw.csv')
  save_to_csv(Zoo, 'data/zo.csv')
  
}

load_datasets()










