const results = [];
const ids = [1, 2, 3];

ids.forEach(async function (id) {
  const res = await fetch("/api/item/" + id);
  results.push(await res.json());
});

console.log("collected", results.length);
