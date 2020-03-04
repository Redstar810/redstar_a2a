/*!
         @file    main.cpp

         @brief

         @author  Hideo Matsufuru (matsufuru)
                  $LastChangedBy: akahoshi $

         @date    $LastChangedDate:: 2019-01-21 17:06:23 #$

         @version $LastChangedRevision: 1929 $
*/

#include "bridge_setup.h"
#ifdef DEBUG
#include "bridge_init_factory.h"
#endif
#include "Parameters/commonParameters.h"
#include "Parameters/parameters.h"
#include "Parameters/parameterManager_YAML.h"
#include "Tools/timer.h"

//#ifdef USE_TESTMANAGER
//#include "run_testmanager.h"
//#endif

#include "IO/bridgeIO.h"
using Bridge::vout;


const std::string filename_main_input = "main.yaml";
// const std::string filename_main_input = "stdin";

//- prototype declarations
int main_core(Parameters*);

//====================================================================
int main(int argc, char *argv[])
{
  // ###  initial setup  ###
  Bridge::VerboseLevel vl = Bridge::GENERAL;

  //- initialization step one
  bridge_initialize(&argc, &argv);
  
  std::string filename_input = filename_main_input;
  
  if (filename_input == "stdin") {
    vout.general(vl, "input filename : ");
    std::cin >> filename_input;
    vout.general(vl, "%s\n", filename_input.c_str());
  } else {
    vout.general(vl, "input filename : %s\n", filename_input.c_str());
  }
  vout.general(vl, "\n");
  
  //- load input parameters (lattice size, MPI grid, etc...)
  Parameters params_all  = ParameterManager::read(filename_input);
  Parameters params_main = params_all.lookup("Main");

  //- initialization step two: setup using parameter values,
  bridge_setup(params_main);

  if(argc == 1){
    vout.general("\nerror: input parameter file is not given.\n");
    vout.general("usage: $%s [Parameter file (hoge.yaml)] \n\n",argv[0]);
    EXIT_FAILURE;
  }
  if(argc == 2){
  
    //- load input parameters (details specific to each configurations)
    std::string filename_conf(argv[1]); // assume the 1st argument is an input filename
    vout.general("\ninput filename (configuration details) : %s\n",filename_conf.c_str());
    Parameters params_conf_all = ParameterManager::read(filename_conf);

#ifdef USE_FACTORY
#ifdef DEBUG
    bridge_report_factory();
#endif
#endif

    //- start timer
    unique_ptr<Timer> timer(new Timer("Main"));
    timer->start();
    
    //- execution main part 
    main_core(&params_conf_all);

    //- find total elapsed time
    timer->report();

  }
  if(argc > 2){
    vout.general("\nerror: too many arguments for %s.\n\n",argv[0]);
    EXIT_FAILURE;
  }

  //- finalization step
  bridge_finalize();

  return EXIT_SUCCESS;
}


//====================================================================
//============================================================END=====
