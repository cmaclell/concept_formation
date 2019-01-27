(function () {
    vt = {};
    vt.CV_KEY = "#ContinuousValue#";

    /*
     * constant - there is only 1 value and it is not numeric
     * constant-numeric - a numeric value where min === max, i.e. no variance.
     * numeric - a numeric value
     * binary - there are only 2 values
     * nominal - there are more than 2 values
     * nominal20 - there are more than 20 values
     * unique - n(values) === n(instances) 
     */
    vt.valType = function(treeRoot, attr) {
        var vals = Object.keys(treeRoot['counts'][attr]);
        if(vals.length === treeRoot['size']) {
          return 'unique';
        }
        if(vals.length > 10) {
          return 'nominal20';
        }
        if(vals.length > 2) {
          return 'nominal';
        }
        if(vals.length > 1) {
          return 'binary';
        }
        if(vals[0] === vt.CV_KEY) {
          return "numeric";
        }
        else {
          return "constant";
        } 
    }

    vt.attrType = function(treeRoot, attr) {
        if (/\('.+'\,.*\)/g.test(attr)) {
            if(/\('_.+'\,.*\)/g.test(attr)){
                return "hidden-relation";
            }
            else if (/\('.+'\,\s+'\?.+'\)/g.test(attr)){
                return "component-relation";
            }
            else {
               return "constant-relation";
            }
        }
        else {
            if (attr[0] === "?") {
                return "variable";
            }
            else if (attr[0] === "_") {
                return "hidden";
            }
            else {
               return "constant";
            }
        }
    }

    vt.filterTypes = function(attr) {
        var typs = []
        if(attr[0] === '_'){
          typs.push('hidden');
        }
        if(/\('.+'\,.*\)/g.test(attr)){
          typs.push('relation');
        }
        if(/\('.+'\,'\?.+'\)/g.test(attr)) {
          typs.push('component');
        }
        return typs;
    }

    vt.min_max = function(node, attr, mm) {
        if(node.counts[attr] === undefined){
            return mm;
        }
        if(node.counts[attr][vt.CV_KEY].mean < mm[0]){
            mm[0] = node.counts[attr][vt.CV_KEY].mean ;
        }
        if(node.counts[attr][vt.CV_KEY].mean  > mm[1]){
            mm[1] = node.counts[attr][vt.CV_KEY].mean ; 
        }
        for (var i = node.children.length - 1; i >= 0; i--) {
            mm = vt.min_max(node.children[i],attr,mm);
        }
        return mm;
    }

    // a bit of a hack to make sure binary variaables that have sensible meaning are ordered correctly
    // the values in each list are based on the test datasets we have and my own thoughts
    vt.isPositiveBinary = function(vals) {
         return $.inArray(vals[0].toLowerCase(), ["success", "yea", "yes", "y", "pro", "pass"]) > -1 &&
            $.inArray(vals[1].toLowerCase(), ["failure", "fail", "nay", "no", "n", "con"]) > -1;
    }

    vt.binaryRatio = function(c, targetAttr, rootObj){
        var rootVals = Object.keys(rootObj['counts'][targetAttr]);
        rootVals.sort();
        rootVals.reverse();
        if (c.counts) { c = c.counts; }
        c = c[targetAttr]
        var values = Object.keys(c);
        values.sort();
        values.reverse();

        //Special case to properly sort a success/failure binary
        if (vt.isPositiveBinary(rootVals)) {
            values.reverse();
            rootVals.reverse();
        }

        if (values.length === 1) {
            return values[0] === rootVals[0] ? 0.0 : 1.0;
        } else {
            return c[values[1]] / (c[values[0]] + c[values[1]]);
        }
    }

    vt.majorityValue = function(c, attr) {
        if (attr === "none") { return ""; }
        if (c.counts) { c = c.counts; }
        var vals = Object.keys(c[attr]);
        if (vals.indexOf(vt.CV_KEY) !== -1){
            return c[attr][CV_KEY].mean;
        } else {
            var maj = {"n":"","v":0};
            for (var i = 0; i < vals.length; i++) {
                if(c[attr][vals[i]] > maj.v){
                    maj.n = vals[i];
                    maj.v = c[attr][vals[i]];
                }
            }
            return maj.n;
        }
    }

    // This is currently not smart enough for continuous values but its also never used with them so it doesn't matter
    vt.randomValue = function(c, attr) {
        if (attr === "none"){ return ""; }
        if (c.counts) { c = c.counts; }
        var vals = Object.keys(c[attr]);
        return vals[Math.floor(Math.random() * vals.length)];
    }

    window.vt = vt;
})();