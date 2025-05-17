[инструкции](https://rmarcus.info/bao_docs/tutorial.html)

заводим сам контейнер с postgres и bao внутри.

добавил имя `bao_postgres` чтобы удобно обращаться
```bash
docker run --name bao_postgres -p 127.0.0.1:5432:5432/tcp --add-host host.docker.internal:host-gateway --shm-size=8g ryanmarcus/imdb_bao:v1
```

До запуска контейнера важно учесть, что если есть желание проверить с gpu, требуется писать такую команду:
``` bash
docker run --gpus=all --name bao_postgres -p 127.0.0.1:5432:5432/tcp --add-host  host.docker.internal:host-gateway --shm-size=8g ryanmarcus/imdb_bao:v1
```

Внутри работает сервер postgres. Чтобы к нему подключиться, через консоль нужно сделать:

``` bash
docker exec -it --user root bao_postgres /bin/bash
```

если не задать пользователя, то подключается к пользователю `postgres`.

Чтобы bao работал, нужно сделать несколько вещей

## 1) запуск сервера с BAO

команды делаем внутри контейнера, через root пользователя.

ставим нормальный питон через миниконду
``` bash
mkdir -p ~/miniconda3 
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
```

теперь появляется python в среде миниконды с pip ом.
устанавливаем нужные пакеты.
``` bash
pip install scikit-learn numpy joblib torch psycopg2-binary
```

устанавливаем `nano`, чтобы редактировать конфиг bao
``` bash
curl https://nano-editor.org/dist/v7/nano-7.2.tar.gz > nano.tar.gz
tar -xvf nano.tar.gz & rm nano.tar.gz
cd nano-7.2/ & ./configure
sudo make & sudo make install
cd .. & rm -rf nano-7.2
```

Перед запуском сервера нужно заменить в train.py все вхождения os.rename на shutil_move.

### запуск bao
``` bash
cd bao_server
python main.py
```

там же у bao лежит файл с конфигами . В нем можно поменять порты по которому сервер бао будет слушать и отвечать. По умолчанию значения выставлены на `ListenOn = '127.0.0.1'  Port=9381` запоминаем.

[описание](https://rmarcus.info/bao_docs/bao_vars.html)

переходим к psql
## 2) postgres
В самом начале нужно разобраться с парой вещей. Во-первых, в файл /etc/postgresql/12/main/pg_hba.conf добавить строки:
```
# IPv4 local connections:
host    all             all             127.0.0.1/32            trust
# IPv6 local connections:
host    all             all             ::1/128                 trust
```

При этом учтите, что нельзя менять у файла разрешение, потому строки добавляются только с помощью root.

После этого перезапустить postgresql с помощью

``` bash
sudo service postgresql reload
```

``` bash
psql -U imdb
```

у нее есть конфиг связанный с bao, можно посмотреть тут [link](https://rmarcus.info/bao_docs/pg_vars.html).
и так распечатать значения
```sql
 SELECT name, setting FROM pg_settings WHERE name LIKE '%bao%';
```

нам нужны `pg_bao.enable_bao, pg_bao.bao_host, pg_bao.bao_port` выставляем их как в bao
``` sql
BEGIN;
SET pg_bao.enable_bao TO on;
SET pg_bao.bao_host TO "127.0.0.1";
SET pg_bao.bao_port TO 9381;
COMMIT;
```

`pg_bao.bao_host` по умолчанию выставлен на какое-то непонятное значение, у меня сработало только когда явно написал.

проверяем 
``` sql
EXPLAIN SELECT count(*) FROM title;
```

### удобства для работы
проблема: чтобы менять код удобно подключиться через расширение докера в vs code, но в нем нельзя редактировать папку `root`  пользователем postgres.
решение: просто выдаешь разрешение всем ее редактировать и не паришься


### Дальнейшие действия

Перед запуском run_queries.py нужно сделать две вещи: добавить папку с sample_queries внутрь root, что проще всего сделать через VS Code, а также поменять немного сам файл run_queries.py

Нужно после строки

``` python
            cur.execute("SET pg_bao.bao_num_arms TO 5")
```

вставить строки

``` python
            cur.execute("SET pg_bao.bao_host TO \"127.0.0.1\"")
            cur.execute("SET pg_bao.bao_port TO 9381")
```

Запуск для проверки работы Bao происходит с помощью команды

``` bash
python3 run_queries.py sample_queries/*.sql | tee ~/bao_run.txt
```

Затем для проверки работы оптимизатора postgresql следует для начала поменять значение переменной USE_BAO на False, после чего выполнить команду

``` bash
python3 run_queries.py sample_queries/*.sql | tee ~/pg_run.txt
```

Анализ работы Bao можно провесити с помощью analyze_bao.ipynb, запуск производится с помощью Jupyter Notebook.