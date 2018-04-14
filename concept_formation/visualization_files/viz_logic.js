(function () {
  const CV_KEY = "#ContinuousValue#"; 

  window.data = {}

  var w = 1280,
    h = window.innerHeight,
    r = 720,
    x = d3.scaleLinear().range([0, r]),
    y = d3.scaleLinear().range([0, r]),
    focus,
    root,
    nodes;

  var svg = d3.select("svg"),
      margin = 20,
      diameter = +svg.attr("viewBox").split(" ")[3];
      g = svg.append("g").attr("transform", "translate("+diameter / 2 + "," + diameter/2 +")");

  var pack = d3.pack()
              .size([diameter - margin, diameter - margin])
              .padding(2);

  var av_properties = {};
  var saturationScale = d3.scaleLinear()
                          .domain([0,1])
                          .range(["#cccccc","steelblue"]);
  var defaultGrey = "#cccccc";

  /*
   * constant - there is only 1 value and it is not numeric
   * constant-numeric - a numeric value where min === max, i.e. no variance.
   * numeric - a numeric value
   * binary - there are only 2 values
   * nominal - there are more than 2 values
   * nominal20 - there are more than 20 values
   * unique - n(values) === n(instances) 
   *
   */
  function valType(rootObj,attr){
    var vals = Object.keys(rootObj['counts'][attr]);

    if(vals.length === rootObj['size']) {
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
    if(vals[0] === CV_KEY) {
      return "numeric";
    }
    else {
      return "constant";
    }
  }

  function attrType(rootObj, attr) {
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

  function filterTypes(attr) {
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

  function min_max(obj, attr, mm) {
    if(obj.counts[attr] === undefined){
      return mm;
    }
    if(obj.counts[attr][CV_KEY].mean < mm[0]){
      mm[0] = obj.counts[attr][CV_KEY].mean ;
    }
    if(obj.counts[attr][CV_KEY].mean  > mm[1]){
      mm[1] = obj.counts[attr][CV_KEY].mean ; 
    }
    for (var i = obj.children.length - 1; i >= 0; i--) {
      mm = min_max(obj.children[i],attr,mm);
    }
    return mm;
  }

  window.init = function(rootObj) {
    av_properties["none"] = {"attrType":"constant","valType":"default","scale":null};
    var attrs = Object.keys(rootObj['counts']);
    for (var i = 0; i < attrs.length; i ++) {
      var attr = attrs[i];
      var vals = Object.keys(rootObj['counts'][attr]);

      var vType = valType(rootObj,attr);
      var aType = attrType(rootObj,attr);
      var at_scale = {"attrType":aType,"valType":vType,"scale":null,"filterTypes":filterTypes(attr)};
      if (vType === "nominal") {
        at_scale.scale = d3.scaleOrdinal(d3.schemeCategory10).domain(vals);
      }
      else if (vType === "nominal20") {
        at_scale.scale = d3.scaleOrdinal(d3.schemeSet3).domain(vals);
      }
      else if (vType === "binary") {
        at_scale.scale = d3.scaleSequential(d3.interpolateViridis);
        at_scale.cat_scale = d3.scaleOrdinal(d3.schemeCategory10).domain(vals);
        at_scale.num_scale = d3.scaleSequential(d3.interpolateViridis);
      }
      else if (vType === "numeric") {
        var mm = min_max(rootObj,attr,[Infinity,-Infinity]);
        if(mm[0] === mm[1]){
          at_scale.valType = "constant-numeric";
          at_scale.scale = saturationScale;
        }
        else {
         at_scale.scale = d3.scaleSequential(d3.interpolateViridis).domain([mm[0],mm[1]]);
        }
      }
      else if (vType === "constant") {
        at_scale.scale = saturationScale;
      }
      av_properties[attr] = at_scale;
    }
  }

  function randomImage(d,attr) {
    if(attr === "none"){
        return "";
    }
    if(d.counts){
        d=d.counts;
    }
    var vals = Object.keys(d[attr]);
    return  "images/" + vals[Math.floor(Math.random() * vals.length)] + ".jpg";
  }

  function hasImageData(d) {
    if(d.counts){
        d=d.counts;
    }
    return d[$("#img-attr").val()];
  }

  function imageSelectChanged() {
    changeImages(data);
  }

  function changeImages(d){
    var img = $("#"+d.name+"-img");
    if(img){
        img.attr("href",randomImage(d,$("#img-attr").val()));
    }
    if(d.children) {
        for(var i in d.children){
          changeImages(d.children[i]);
        }
    }
  }

  // A lot of this function is based on the Zoomable Circle Packing Example:
  // https://bl.ocks.org/mbostock/7607535
  window.buildTree = function(tree_data) {

    root = d3.hierarchy(tree_data)
            .sum(function(d) {return d.size; })
            .sort(function(a,b) { return b.value - a.value; });
    
    focus = root;
    nodes = pack(root).descendants();
    var view;

    var circle = g.selectAll("circle")
      .data(nodes)
      .enter().append("circle")
        .attr("id",function(d) {return d.data.name; })
        .attr("class", function(d) { return d.children ? "parent" : "child"; })
                                                                  //This is a stupid hack
        .on("click", function(d) { if (focus !== d) zoom(d), d3.event ? d3.event.stopPropagation() : undefined });

    // I Removed images for the time being while updating to newer d3 code.
    // var images = g.selectAll("image.screenshot")
    //   .data(nodes.filter(function(d){
    //         if (d.children){// || !hasImageData(d)){ 
    //           return false; 
    //         } else {
    //           return true;
    //         }}))
    //   .enter().append("image")
    //       .attr("id",function(d) { return d.name+"-img";})
    //       .attr('class', 'screenshot')
    //       .attr("xlink:href", function(d) { return randomImage(d,$("#img-attr").val());})
    //   .attr("width", function(d) { return (1.33 * d.r); })
    //   .attr("height", function(d) { return d.r; })
    //   .attr("x", function(d) { return d.x - ((1.33 * d.r)/2); })
    //   .attr("y", function(d) { return d.y - (d.r/2); })
    //   .on("click", function(d) { return zoom(node == d ? root : d); });

    var text = g.selectAll("text")
      .data(nodes)
      .enter().append("text")
        .attr("class","label")
        .style("fill-opacity", function(d) { return d.parent === root ? 1 : 0; })
        .style("display", function(d) { return d.parent === root ? "inline" : "none"; })
        .text(function(d) { return d.data.name; });

    var node = g.selectAll("circle,text");

    svg.on("click", function() { zoom(root); });

    zoomTo([root.x, root.y, root.r * 2 + margin]);

    window.zoom = function(d) {
      // console.log(d)
      make_property_sheet(d.data);
      var focus0 = focus; focus = d;

      var transition = d3.transition()
          .duration(d3.event && d3.event.altKey ? 7500 : 750)
          .tween("zoom", function(d) {
            var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
            return function(t) { zoomTo(i(t)); };
          });

      transition.selectAll("text")
        .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
          .style("fill-opacity", function(d) { return d.parent === focus ? 1 : 0; })
          .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
          .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });
    }

    function zoomTo(v) {
      var k = diameter / v[2]; view = v;
      node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
      circle.attr("r", function(d) { return d.r * k; });
    }

    // changeText();  
  }

  function majorityValue(table) {
    var values = Object.keys(table);
    values.sort();
    values.reverse();
    var maj = {"n":"","v":0};
    for (var i = 0; i < values.length; i++) {
      if(table[values[i]] > maj.v){
        maj.n = values[i];
        maj.v = table[values[i]];
      }
    }
    return maj.n;
  }

  // a bit of a hack to make sure binary variaables that have sensible meaning are ordered correctly
  // the values in each list are based on the test datasets we have and my own thoughts
  function isPositiveBinary(vals) {
    return $.inArray(vals[0].toLowerCase(), ["success","yea","yes","y","pro","pass"]) > -1 &&
          $.inArray(vals[1].toLowerCase(), ["failure", "fail","nay","no","n","con"]) > -1;
  }

  function binaryRatio(table,targetAttr) {
    var rootVals = Object.keys(data['counts'][targetAttr]);
    rootVals.sort();
    rootVals.reverse();
    var values = Object.keys(table);
    values.sort();
    values.reverse();

    //Special case to properly sort a success/failure binary
    if(isPositiveBinary(rootVals)){
      values.reverse();
      rootVals.reverse();
    }

    if(values.length === 1) {
      return values[0] === rootVals[0] ? 0.0 : 1.0;
    }
    else {
      return table[values[1]]/(table[values[0]]+table[values[1]]);
    }
  }

  function recolor(d) {
    // console.log(d)
    var targetAttr = $("#focus-attr").val();
    var scale = av_properties[targetAttr];
    if(!scale){
      console.log("No scale found for attribute:" + targetAttr);
      return;
    }
    var circle = $("#"+d.name);
    // If the concept doesn't have the target attribute make it grey
    if (targetAttr !== "none" && !d['counts'][targetAttr]) {
      circle.css("fill",defaultGrey);
    }
    else {
      switch(scale.valType) {
        case "constant":
          var val = Object.keys(d['counts'][targetAttr])[0];
          circle.css("fill",scale.scale(d['counts'][targetAttr][val]/d['size']));
          break;

        case "constant-numeric":
          circle.css("fill",scale.scale(d['counts'][targetAttr][CV_KEY].n/d['size']));
          break;
        
        case "nominal":
        case "nominal20":
          circle.css("fill",scale.scale(majorityValue(d['counts'][targetAttr])));
          break;
        
        case "binary":
          if($("#binary-style").val() === "categorical") {
            circle.css("fill",scale.cat_scale(majorityValue(d['counts'][targetAttr],targetAttr)));
          }
          else {
           circle.css("fill",scale.num_scale(binaryRatio(d['counts'][targetAttr],targetAttr))); 
          }
          break;
        
        case "numeric":
          var num = d['counts'][targetAttr][CV_KEY];
          circle.css("fill",scale.scale(num.mean));
          break;
        
        case "unique":
        default:
          if(d.children.length > 0) {
            circle.css("fill","steelblue");
          }
          else {
            circle.css("fill",defaultGrey);
          }
          break;
      }
    }
    if(d.children){
        for(var child in d.children){
          recolor(d.children[child]);
        }
    }
  }

  window.populateColorByOptions = function(d) {
    var attrs = Object.keys(d['counts']);
    attrs.sort();
    attrs.reverse();
    var colorBy = $("#focus-attr");
    var imgOptions = $("#img-attr");
    colorBy.html('<option value="none">None</option>');

    for (var i=0; i < attrs.length;i++){
      var attrScale = av_properties[attrs[i]];
      
      var opt = $("<option>"+attrs[i]+"</option>");
      opt.attr("value",attrs[i]);
      imgOptions.append(opt.clone());

      if(attrScale) {
        if($("#fil-val-unique").is(':checked') && attrScale.valType === "unique") {
          continue;
        }
        else if($("#fil-val-constant").is(':checked') && (attrScale.valType === "constant"|| attrScale.valType === "constant-numeric")) {
          continue;
        }
        else if($("#hid-filter").is(':checked') && attrScale.filterTypes.indexOf("hidden") !== -1){
          continue;
        }
        else if( $("#com-filter").is(':checked') && attrScale.filterTypes.indexOf("component") !== -1) {
          continue;
        }
        else if($("#rel-filter").is(':checked') && attrScale.filterTypes.indexOf("relation") !== -1) {
          continue;
        }
      }
      colorBy.append(opt);
    }
    colorBy.change(colorSelectChanged);
  }

  window.colorSelectChanged = function() {
    var attr = $("#focus-attr").val();
    if(attr === "none"){
      $("#focus-attr-type").text("Default");
      $("#focus-val-type").text("Default");
      $(".color-ops").hide();
    }
    else {
      var attrScale = av_properties[attr];
      switch(attrScale.valType){
        case "unique":
          $("#focus-val-type").text("Unique");
          $("#numeric-colors").hide();
          $("#nominal-colors").hide();
          break;
        case "constant":
        case "constant-numeric":
          $("#focus-val-type").text("Constant"); 
          $("#numeric-colors").hide();
          $("#nominal-colors").hide();
          break;
        case "nominal":
        case "nominal20":
          $("#focus-val-type").text("Nominal");
          var colorTable = $("#nom-val-colors");
          colorTable.html('');
          var vals = Object.keys(data['counts'][attr]);
          vals.sort();
          vals.reverse();
          for(i=0;i<vals.length;i++) {
            var tr = $("<tr></tr>");
            var valName = $("<td>"+ vals[i] +"</td>");
            var colorCell = $("<td></td>");
            valName.addClass("absorbing-column");
            colorCell.addClass("swatch");
            colorCell.css("background-color",attrScale.scale(vals[i]));
            tr.append(valName);
            tr.append(colorCell);
            colorTable.append(tr);
          }
          $("#numeric-colors").hide();
          $("#nominal-colors").show();
          break;
        case "numeric":
          $("#focus-val-type").text("Numeric");
          var dom = attrScale.scale.domain();
          $("#num-val-1").text(dom[0].toFixed(3));
          $("#num-val-9").text(dom[1].toFixed(3));
          $("#num-val-5").text(((dom[1]+dom[0])/2).toFixed(3));
          for (var i = 0; i < 10; i++) {
            $("#num-col-"+i).css("background-color",attrScale.scale(dom[0] + (dom[1]-dom[0])/9*i));
          }
          $("#numeric-colors").show();
          $("#nominal-colors").hide();
          break;
        case "binary":
          $("#focus-val-type").text("Binary");
          var rootVals = Object.keys(data['counts'][attr]);
          rootVals.sort();
          rootVals.reverse();
          if(isPositiveBinary(rootVals)){
            rootVals.reverse();
          }
          if($("#binary-style").val() === "categorical"){
            var colorTable = $("#nom-val-colors");
            colorTable.html('');
            for(i=0;i<rootVals.length;i++) {
              var tr = $("<tr></tr>");
              var valName = $("<td>"+ rootVals[i] +"</td>");
              var colorCell = $("<td></td>");
              valName.addClass("absorbing-column");
              colorCell.addClass("swatch");
              colorCell.css("background-color",attrScale.cat_scale(rootVals[i]));
              tr.append(valName);
              tr.append(colorCell);
              colorTable.append(tr);
            }
            $("#numeric-colors").hide();
            $("#nominal-colors").show();
          }
          else {
            $("#num-val-1").text(rootVals[0]);
            $("#num-val-9").text(rootVals[1]);
            $("#num-val-5").text("");
            for (var i =0; i < 10; i++){
              $("#num-col-"+i).css("background-color",attrScale.num_scale(i/9));
            }
            $("#numeric-colors").show();
            $("#nominal-colors").hide();
          }
          break;
        default:
          console.error("unknown attribute type found");
          console.log(data);
      }
      switch(attrScale.attrType){
        case "hidden-relation":
          $("#focus-attr-type").text("Hidden Relation");
          break;
        case "component-relation":
          $("#focus-attr-type").text("Component Relation");
          break;
        case "constant-relation":
          $("#focus-attr-type").text("Constant Relation");
          break;
        case "variable":
          $("#focus-attr-type").text("Variable");
          break;
        case "hidden":
          $("#focus-attr-type").text("Hidden");
          break;
        case "constant":
        default:
          $("#focus-attr-type").text("Constant");
          break;
      }
    }
    recolor(data);
  }

  window.make_property_sheet = function(node_data) {
    d = node_data['counts']
    
    var property_sheet = $("#properties"); 
    var prop_filter = $("#property-filter");
    property_sheet.html('');

    $("#concept-name").text(node_data['name'])
    $("#concept-size").text(node_data['size'])
    var n_children = node_data['children'] ? node_data['children'].length : 0;
    $("#concept-child-size").text(n_children)
    var d_keys = Object.keys(d);
    var attrs = []

    for (var i = 0; i < d_keys.length; i++){
      var attrScale = av_properties[d_keys[i]];

      if(attrScale) {
        if($("#fil-val-unique").is(':checked') && attrScale.valType === "unique") {
          continue;
        }
        if($("#fil-val-constant").is(':checked') && (attrScale.valType === "constant" || attrScale.valType === "constant-numeric")) {
          continue;
        }
        if($("#fil-val-binary").is(':checked') && attrScale.valType === "binary") {
          continue;
        }
        if($("#fil-val-numeric").is(':checked') && attrScale.valType === "numeric") {
          continue;
        }
        if($("#fil-val-nominal").is(':checked') && (attrScale.valType === "nominal" || attrScale.valType === "nominal20")) {
          continue;
        }
        if($("#hid-filter").is(':checked') && attrScale.filterTypes.indexOf("hidden") !== -1){
          continue;
        }
        if($("#com-filter").is(':checked') && attrScale.filterTypes.indexOf("component") !== -1) {
          continue;
        }
        if($("#rel-filter").is(':checked') && attrScale.filterTypes.indexOf("relation") !== -1) {
          continue;
        }
        attrs.push(d_keys[i]);
      }

    }
    attrs.sort();
    attrs.reverse();
    for (var a in attrs) {
      attr = attrs[a]
      var tr = $("<tr></tr>");
      var ar = $('<td colspan="3">'+attr+'</td>');
      ar.addClass("prop-column")
      tr.addClass("info");
      tr.append(ar);
      property_sheet.append(tr);
      vals = Object.keys(d[attr]);
      vals.sort();
      vals.reverse();
      for (var v in vals){
        val = vals[v];
        tr = $("<tr></tr>");
        var vd = $("<td></td>");
        var n = NaN;
        //if this is a continuous value
        if(val === CV_KEY){
          vd.text(d[attr][val].mean.toFixed(3) + " (" + d[attr][val].std.toFixed(3)+ ")");
          n = d[attr][val].n;
        }
        //else
        else {
          vd.text(val);
          n = d[attr][val];
        }
        vd.addClass("val-row");
        tr.append(vd);
        var cd = $("<td>"+n+"</td>");
        var pd = $("<td>"+(n/node_data['size']*100).toFixed(2)+"%</td>")
        tr.append(cd);
        tr.append(pd);
        property_sheet.append(tr);
      }
    }
  }

  window.updateFilters = function() {
    populateColorByOptions(data);
    colorSelectChanged();
    make_property_sheet(focus.data);
  }

  window.searchConceptByName = function(e){
    var conc_name = $("#search-concept-name").val();
    if(conc_name.length === 0) {
      $(".focus").removeClass("focus");
      $("#search-not-found").hide();
      return;
    }

    if(!conc_name.startsWith("Concept")) {
      conc_name = "Concept" + conc_name;
    }
    
    if (/Concept\d+/g.test(conc_name)){
      if($("#"+conc_name).length === 0) {
        $("#search-not-found").text("No "+conc_name);
        $("#search-not-found").show();
      }
      else {
        $("#search-not-found").hide();
        $(".focus").removeClass("focus");
        var n = undefined;
        for (var i = nodes.length - 1; i >= 0; i--) {
         if (nodes[i].data.name == conc_name) {
          n = nodes[i];
         }
        }
        if(n!==undefined) g.select("#"+conc_name).on("click")(n);
        $("#"+conc_name).addClass("focus"); 
      }
    }  
  }
})();