"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
    ** Domain * /;
domain ?  : string;
    ** Output;
format * /;
outputFormat ?  : 'triples' | 'entities' | 'graph';
    ** Nodes;
accessed * /;
nodesAccessed ?  : number;
relationshipsTraversed ?  : number;
indexesUsed ?  : string[];
    ** Strategy;
used * /;
strategy: string;
    ** Model;
version * /;
modelVersion ?  : string;
    ** Document;
statistics * /;
statistics: {
    wordCount: number;
    sentenceCount: number;
    paragraphCount: number;
}
;
//# sourceMappingURL=knowledge.js.map