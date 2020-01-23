% Main: runs batches of data catagories in our case these are genres
% Produces a text file with the analysis of each genre (recall, precision,
% confusion matrix etc. see evaluation for more info.
function main(genres)
    fileID = fopen("overall_results.txt",'w');
    fprintf(fileID,"Spotify-NEAT Results\n");
    fclose(fileID);
    for genre = genres
        neat_main(genre) %evolution of population
        saveas(gcf,"graphs/"+genre+".png") %save graph of resulting evolution
        
        fileID = fopen("overall_results.txt",'a');
        fprintf(fileID,genre+"\n");
        fclose(fileID);
        
        evaluation(genre); % generate further analysis
    end
end