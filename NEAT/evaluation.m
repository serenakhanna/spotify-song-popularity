% Evaluate provides an evaluation of a given neat save this includes:
% - Graphical representation of the fittest genotype
% - Confusion matrix of the fittest genotype
% - Presision, Recall of the fittest
% - Average fitness

function error = evaluation(genre)
    load("neatsave_"+genre+".mat");
    
    population_plus_fitnesses=population;
    no_change_threshold=1e-3; %threshold to judge if state of a node has changed significantly since last iteration
    number_individuals=size(population,2);
    
    dataset=csvread("datasets/"+genre+".csv");
    [m,n] = size(dataset);
    
    input_pattern = dataset(:,1:(n-1));

    [total_patterns,input_nodes]=size(input_pattern);

    output_pattern = dataset(:,n);
    
    confusion = zeros(2,2);
    
    % Find the Fittest
    max_fit_individual = 1;
    max_fitness = 0;
    for index_individual=1:number_individuals
        if(population(index_individual).fitness > max_fitness)
            max_fitness = population(index_individual).fitness;
            max_fit_individual = index_individual;
        end
    end
   display("Max Fitness:"+num2str(max_fitness)+" Individual:"+num2str(max_fit_individual));
    
   % Evaluate the network of Fittest:
   number_nodes=size(population(max_fit_individual).nodegenes,2);
   number_connections=size(population(max_fit_individual).connectiongenes,2);
   individual_fitness=0;
   output=[];
   hidden_nodes = (input_nodes+2):number_nodes;

   for index_pattern=1:total_patterns
      % the following code assumes node 1 to 13 inputs, node 14 bias, node 15 output, rest arbitrary (if existent, will be hidden nodes)
      % set node input steps for first timestep
      population(max_fit_individual).nodegenes(3,hidden_nodes)=0; %set all node input states to zero
      population(max_fit_individual).nodegenes(3,input_nodes+1)=1; %bias node input state set to 1
      population(max_fit_individual).nodegenes(3,1:input_nodes)=input_pattern(index_pattern,:); %node input states of the two input nodes are consecutively set to the given dataset input pattern  

      %set node output states for first timestep (depending on input states)
      population(max_fit_individual).nodegenes(4,1:(input_nodes+1))=population(max_fit_individual).nodegenes(3,1:(input_nodes+1));
      population(max_fit_individual).nodegenes(4,hidden_nodes)=1./(1+exp(-4.9*population(max_fit_individual).nodegenes(3,hidden_nodes)));
      no_change_count=0;     
      index_loop=0;
      
      % main evaluation
      while (no_change_count<number_nodes) & index_loop<3*number_connections
         index_loop=index_loop+1;
         vector_node_state=population(max_fit_individual).nodegenes(4,:);
         for index_connections=1:number_connections
            %read relevant contents of connection gene (ID of Node where connection starts, ID of Node where it ends, and connection weight)
            ID_connection_from_node=population(max_fit_individual).connectiongenes(2,index_connections);
            ID_connection_to_node=population(max_fit_individual).connectiongenes(3,index_connections);
            connection_weight=population(max_fit_individual).connectiongenes(4,index_connections);
            %map node ID's (as extracted from single connection genes above) to index of corresponding node in node genes matrix
            index_connection_from_node=find((population(max_fit_individual).nodegenes(1,:)==ID_connection_from_node));
            index_connection_to_node=find((population(max_fit_individual).nodegenes(1,:)==ID_connection_to_node));

            if population(max_fit_individual).connectiongenes(5,index_connections)==1 %Check if Connection is enabled
               population(max_fit_individual).nodegenes(3,index_connection_to_node)=population(max_fit_individual).nodegenes(3,index_connection_to_node)+population(max_fit_individual).nodegenes(4,index_connection_from_node)*connection_weight; %take output state of connection_from node, multiply with weight, add to input state of connection_to node
            end
         end
         %pass on node input states to outputs for next timestep 
         population(max_fit_individual).nodegenes(4,hidden_nodes)=1./(1+exp(-4.9*population(max_fit_individual).nodegenes(3,hidden_nodes)));          
         %Re-initialize node input states for next timestep
         population(max_fit_individual).nodegenes(3,hidden_nodes)=0; %set all output and hidden node input states to zero
         no_change_count=sum(abs(population(max_fit_individual).nodegenes(4,:)-vector_node_state)<no_change_threshold); %check for alle nodes where the node output state has changed by less than no_change_threshold since last iteration through all the connection genes
      end      

      output=[output;population(max_fit_individual).nodegenes(4,(input_nodes+2))];
      
      
      individual_fitness=individual_fitness+abs(output_pattern(index_pattern,1)-population(max_fit_individual).nodegenes(4,(input_nodes+2))); %prevent oscillatory connections from achieving high fitness   

%       display([output_pattern(index_pattern,1),population(max_fit_individual).nodegenes(4,15)]);

   end
   error = individual_fitness/total_patterns;
    
   
   

%    display("Fitness: "+(individual_fitness))
   population_plus_fitnesses(max_fit_individual).fitness=(m-individual_fitness)^2; %Fitness function as defined by Kenneth Stanley, CONFUSED IF 15 IS THE CORRECT NUMBER
   output = round(output);
   
   % Confusion Matrix
   displacement = round(output)-output_pattern;
   confusion(1,2) = size(find(displacement==-1),1);
   confusion(2,1) = size(find(displacement==1),1);
   
   truths = find(displacement==0);
   for truth = truths'
       output(truth);
       if(output(truth)==0)
           confusion(2,2) = confusion(2,2) + 1;
       end
       if(output(truth)==1)
           confusion(1,1) = confusion(1,1) + 1;
       end
   end
   confusion
   
   fileID = fopen("overall_results.txt",'a');
   fprintf(fileID,"Fittest Individual: %d , Fitness: %d\nConfusion:\n",max_fit_individual,individual_fitness);
   fprintf(fileID,"%5d %5d\n",confusion');
   fclose(fileID);
   
   
   
   % Recall
   col_sum = sum(confusion,1); % recall
   row_sum = sum(confusion,2); % precision
   
   % Success:
   recall_1 = confusion(1,1)/col_sum(1);
   recall_0 = confusion(2,2)/col_sum(2);
   
   
   % Precision
   precision_1 = confusion(1,1)/row_sum(1);
   precision_0 = confusion(2,2)/row_sum(2);
   
   fileID = fopen("overall_results.txt",'a');
   fprintf(fileID,"Recall:\n-Success: %d \n-Failure: %d \nPrecision:\n-Success: %d \n-Failure: %d\n",recall_1,recall_0,precision_1,precision_0);
   fclose(fileID);
   
   
   
end
