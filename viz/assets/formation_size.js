export function matchingShape(value, count) {
  const match = String(value ?? '').match(/^\s*(\d+)\s*[xX×*]\s*(\d+)\s*$/);
  if (!match) return null;
  const columns = Number(match[1]), rows = Number(match[2]);
  return Number.isSafeInteger(columns) && Number.isSafeInteger(rows)
    && columns > 0 && rows > 0 && columns * rows === count ? `${columns}x${rows}` : null;
}

function suggestedShape(count) {
  let rows = Math.floor(Math.sqrt(count));
  while (count % rows !== 0) rows--;
  return `${count / rows}x${rows}`;
}

export function shapeForCount(shape,count) {
  return matchingShape(shape,count) || suggestedShape(count);
}

// Update the model and both controls together, before notifying the JSON preview.
export function bindFormationSize(cluster, countInput, shapeInput, onChange, {synthetic = true} = {}) {
  countInput.min = '1';
  countInput.step = '1';
  shapeInput.title = '修改卫星数时自动匹配尺寸；也可手动输入行×列，乘积须等于卫星数。';

  function validateShape() {
    const count = cluster.satellites_per_cluster;
    const shape = matchingShape(shapeInput.value, count);
    if (synthetic && shape) cluster.grid_shape = shape;
    shapeInput.setCustomValidity(!synthetic || shape ? '' : `初始尺寸的行数×列数须等于每 cell 卫星数（当前为 ${count} 颗）。`);
  }

  function resize() {
    const count = cluster.satellites_per_cluster;
    if (synthetic && Number.isSafeInteger(count) && count > 0) {
      cluster.grid_shape = shapeForCount(cluster.grid_shape, count);
      shapeInput.value = cluster.grid_shape;
    }
    validateShape();
  }

  countInput.oninput = countInput.onchange = () => {
    cluster.satellites_per_cluster = Number(countInput.value);
    resize();
    onChange();
  };
  shapeInput.oninput = shapeInput.onchange = () => {
    cluster.grid_shape = shapeInput.value;
    validateShape();
    onChange();
  };
  // Also repair older drafts whose count and automatically derived shape disagree.
  resize();
}
