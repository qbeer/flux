# Design Doc for Flux

![image](flux_design1.png)

## Flux Database

- Create a general format for data storage.  (For now, tf records)

## Dataset Class

- For general implementation, look at [dataset.py](../datasets/dataset.py)

## Data Incorporation Flow
1. Input:  Download link and Download Format --> Download to empty work folder
2. Download Type
    - Need Process
        - zip, tar, file
    - Don't need process
        - single file
3. Alternate Download?  
    - What if the internet is broken inamist of a big download?  
        - TODO: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch11s06.html
        - use a tmp folder to save undownloaded data
    - Or download to work folder manually
4. For all download, maybe_download should return an processed folder as a path

5. Add the data to datastore
    - root_key should be the header of the path.  i.e. coco2014/data
    - copy the contents in the work folder to path extended by root key
        
#### TODO:
- Implement sample for each dataset

### Vision
- cifar
- mnist

### NLP
- newslens
- squad
