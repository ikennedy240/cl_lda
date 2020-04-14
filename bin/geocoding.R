#where we need to extract addresses from the listed text
arcpyGeocode <- function(tbl, run_fields = NULL, run_no = NULL, debug = FALSE){
  
  #save some values for the geocoding script call
  LISTSRC <- 'kctax'
  LISTLOC <- 'special'
  PYPATH <- "C:/Python27/ArcGISx6410.6/python" 
  PATH <- "Z:/GIT/natrent" 
  geocode_script <- paste0(PATH, "/python/arcpyGeocode.py")
  input_tbl <- file.path('Z:/GIT/cl_lda','resources','tmp.csv')
  output_shp <- file.path('Z:/GIT/cl_lda','resources','kctax_geocoded_2')
  
  #make sure we have the right arguments to proceed
  if(is.null(run_fields)){
    stop("Need to specify the address, city and state fields as ordered character vector.")
  }
  
  #make sure the run_field coumns are character to avoid join errors
  tbl[unique(run_fields)] <- sapply(tbl[unique(run_fields)], as.character)
  
  #reduce the table to unique combinations to be geocoded
  geocode_tbl <- tbl %>%
    select_(run_fields[1], run_fields[2], run_fields[3]) %>%
    distinct()
  
  #make sure the config's PYPATH is appropriate for ArcPy
  if(!grepl("x64", PYPATH)){
    stop("Need to use 64-bit Python and ArcGIS Geoprocessing Tools")
  }
  
  #write the geocode table to storage with the corresponding name
  write.csv(geocode_tbl, input_tbl, row.names = FALSE)
  
  #give us three tries to get past ArcGIS gremlines
  gc_tries <- 1
  gc_result <- NULL
  sleep_time <- 20
  
  #Python2.7 script to use ArcPy.GeocodeAddresses_geocoding
  while(gc_tries <= 3 && is.null(gc_result)){
    gc_result <- try(system(paste(PYPATH, geocode_script, input_tbl, output_shp, 
                                  run_fields[1], run_fields[2], run_fields[3])))
    
    if(inherits(gc_result, "try-error")){
      
      #a lot of observed bugs result from what looks like licensing issues
      #i.e. ArcGIS thinks we are using more licenses than we are allowed
      Sys.sleep(sleep_time)
      
      #reset the gc_result object for another run, mark that we used an attempt
      gc_result <- NULL
      gc_tries <- gc_tries + 1
    }
  }
  
  #read in the point data shapefile that was produced
  outcome <- try(read_sf(paste0(output_shp, ".shp"),
                         stringsAsFactors = FALSE),
                 silent = T)
  
  #end function call based on whether outcome exists / was unsuccessful
  if(!inherits(outcome, "try-error")){
    
    #if it does exist, we only need the tabular data
    outcome$geometry <- NULL
    
    #make sure the run_field coumns are character to avoid join errors
    outcome[unique(run_fields)] <- sapply(outcome[unique(run_fields)], as.character)
    
    #read_sf interprets " " as NA, so switch back to still match to the input tbl correctly
    #if dummy is present
    if("dummy" %in% run_fields){outcome$dummy <- rep(" ", nrow(outcome))}
    
    #silently join this back to the geocoding table that was input
    geocode_tbl <- suppressWarnings(suppressMessages(left_join(geocode_tbl, outcome)))
    
    #now join this back to the original, undeduplicated table
    tbl <- suppressWarnings(suppressMessages(left_join(tbl, geocode_tbl)))
    
    # fill missing 
    tbl <- tbl %>% mutate(RegionAbbr =  if_else(!is.na(RegionAbbr), 
                                                map_chr(Region,function(x) if(!is.na(x)&x %in% state.name) `names<-`(state.abb, state.name)[[x]] else NA),
                                                RegionAbbr))
    #return the tbl that now has geocode fields
    tbl
    
    #if we had three unsuccessful runs and/or couldn't read in the shp we expected
  } else{
    
    #stop and print the last observed error
    stop(paste("ArcGIS geocoding was stopped after 3 unsuccessful attempts.\n\nLast Error:\n", 
               as.character(outcome)))
  }
}

smartyGeoparse <- function(tbl, debug = FALSE){
  
  #set the arguments to pass via cmd
  LISTSRC <- ifelse(config$SOURCE == "Craigslist", "cl", NA)
  LISTLOC <- config$LOCABB
  PYPATH <- config$PYPATH
  geoparse_script <- paste0(config$PATH, "/python/addressExtraction.py")
  input_tbl <- paste0(config$PATH, "/data/geo/Smarty/", LISTSRC, "_", LISTLOC, "_to_geoparse_smarty.csv")
  output_tbl <- paste0(config$PATH, "/data/geo/Smarty/", LISTSRC, "_", LISTLOC, "_craigslist_data_processed.csv")
  
  #if there are files from failed runs in the Smarty working folders, remove them
  file.remove(Sys.glob(paths = paste0(config$PATH, "/data/geo/Smarty/", LISTSRC, "_", LISTLOC, "*")))
  
  #save the temp table to storage
  write_csv(tbl, input_tbl)
  
  #announce we are sending flat file to addressExtraction.py
  cat("Starting API calls to Smartystreets for address validation/extraction...\n\n")
  
  #Python2.7 script to use Smartystreets Python SDK
  system(command=paste(PYPATH, geoparse_script, input_tbl, output_tbl, LISTSRC, LISTLOC))
  
  #read in the result
  outcome <- read_csv(paste0(config$DATAPATH, output_tbl), col_types = cols())
  
  #rename columns related to geocoding
  outcome <- outcome %>%
    rename(match_type = address_category, 
           lng = lon,
           match_address = l1,
           match_address2 = l2) %>%
    #need to make the zipcode 5 digits long since to match Arc format
    mutate(zip5 = substr(str_extract(match_address2, '(?<!\\d)\\d{5}(?:[ -]\\d{4})?\\b'), 1, 5),
           match_address2 = str_replace(match_address2, pattern = '(?<!\\d)\\d{5}(?:[ -]\\d{4})?\\b', zip5),
           program_iteration = as.character(program_iteration),
           temp_add = paste(match_address, match_address2),
           dummy = "<NA>") %>%
    select(-zip5)
  
  #implement a call to arcpyGeocode() to improve on the coordinate precision
  arc_processed <- arcpyGeocode(outcome, 
                                run_fields = c("temp_add", "dummy", "dummy"),
                                run_no = 5)
  
  #clean the temporary files
  file.remove(Sys.glob(paths = paste0(config$PATH, "/data/geo/ESRI/", LISTSRC, "_", LISTLOC, "*")))
  
  #return the outcome table
  arc_processed
}
