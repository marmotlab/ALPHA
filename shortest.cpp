#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int THRESH = 999999;

void printMat(vector<vector<int>> matrix){
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    
}

int checkVal(int first, int second, int ROW, int COL){
    return (first<ROW && first>=0 && second<COL && second>=0) ? 1 : 0;
}

vector<vector<int>> oneRobot(vector<vector<int>> map, pair<int,int> start){
    int ROW = map.size();
    int COL = map[0].size();

    if(!checkVal(start.first, start.second, ROW, COL) || map[start.first][start.second]==-1){
        throw invalid_argument("Invalid Start");
    }


    queue<pair<int,int>> openList;
    vector<vector<int>> distMat(ROW, vector<int>(COL,THRESH));


    distMat[start.first][start.second] = 0;

    openList.push(start);

    while(!openList.empty()){
        pair<int,int> current = openList.front();
        openList.pop();

        if(checkVal(current.first+1, current.second, ROW, COL) && (distMat[current.first+1][current.second]==THRESH) && (map[current.first+1][current.second]!=-1)){
            distMat[current.first+1][current.second] = distMat[current.first][current.second]+1;
            openList.push(pair<int,int>(current.first+1, current.second));
        }

        if(checkVal(current.first-1, current.second, ROW, COL) && (distMat[current.first-1][current.second]==THRESH) && (map[current.first-1][current.second]!=-1)){
            distMat[current.first-1][current.second] = distMat[current.first][current.second]+1;
            openList.push(pair<int,int>(current.first-1, current.second));
        }

        if(checkVal(current.first, current.second+1, ROW, COL) && (distMat[current.first][current.second+1]==THRESH) && (map[current.first][current.second+1]!=-1)){
            distMat[current.first][current.second+1] = distMat[current.first][current.second]+1;
            openList.push(pair<int,int>(current.first, current.second+1));
        }

        if(checkVal(current.first, current.second-1, ROW, COL) && (distMat[current.first][current.second-1]==THRESH) && (map[current.first][current.second-1]!=-1)){
            distMat[current.first][current.second-1] = distMat[current.first][current.second]+1;
            openList.push(pair<int,int>(current.first, current.second-1));
        }
    }

    
    return distMat;
    
}

vector<vector<vector<int>>> extCall(vector<vector<int>> map, vector<pair<int,int>> starts){
    vector<vector<vector<int>>> dist_map;
    for (int i = 0; i < starts.size(); i++)
    {
        dist_map.push_back(oneRobot(map,starts[i]));
    }
    return dist_map;
    
}

int main(int argi, char* argv[]){
    
    vector<vector<int>>map = {{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},{-1,0,0,0,0,-1,0,0,0,0,0,0,0,-1},{-1,0,0,0,0,-1,0,0,0,0,4,0,0,-1},{-1,0,0,0,0,-1,7,0,0,0,0,0,0,-1},{-1,0,0,0,0,0,0,1,0,0,0,0,0,-1},{-1,0,0,0,0,-1,0,0,0,0,0,0,0,-1},{-1,0,0,0,0,-1,0,0,0,0,0,0,0,-1},{-1,0,0,0,0,-1,0,0,0,0,0,0,0,-1},{-1,0,0,5,0,-1,0,0,2,0,0,0,0,-1},{-1,0,0,6,0,-1,0,0,0,0,0,3,0,-1},{-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},{-1,0,0,0,0,0,0,0,0,0,0,0,0,-1},{-1,8,0,0,0,-1,0,0,0,0,0,0,0,-1},{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}};
    vector<pair<int,int>> starts{pair<int,int>(1,1), pair<int,int>(1,2)};

    printMat(map);
    // vector<vector<int>> temp = oneRobot(map,start);
    // printMat(temp);
    vector<vector<vector<int>>> dist_mat = extCall(map, starts);

    for (int i = 0; i < starts.size(); i++)
    {
        printMat(dist_mat[i]);
        printf("\n");
    }
    

    return 0;
}
