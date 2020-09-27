wget http://data.allenai.org/rule-reasoning/rule-reasoning-dataset-V2020.2.4.zip .
unzip rule-reasoning-dataset-V2020.2.4.zip
rm rule-reasoning-dataset-V2020.2.4.zip
mv rule-reasoning-dataset-V2020.2.4 data
cp data_auxilliary/*.tsv data/depth-3ext-NatLang/